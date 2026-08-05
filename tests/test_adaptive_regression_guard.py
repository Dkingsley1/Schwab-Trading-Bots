import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import adaptive_regression_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_guarded_paper_health_fast(project_root: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        project_root / "governance" / "health" / "health_fast_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {
                    "ok": False,
                    "status": "blocked_read_only",
                    "blockers": ["live_execution_requires_explicit_operator_control"],
                },
            },
            "runtime_pressure": {
                "overall_status": "advisory",
                "host_saturation_score": 18.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
            },
        },
    )


def _seed_ready_artifacts(project_root: Path, *, high_pressure: bool = False) -> None:
    health = project_root / "governance" / "health"
    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        health / "health_fast_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "runtime_pressure": {
                "overall_status": "advisory" if high_pressure else "ready",
                "host_saturation_score": 61.0 if high_pressure else 12.0,
                "compute_pressure_level": "high" if high_pressure else "normal",
                "memory_pressure_level": "normal",
            },
        },
    )
    _write_json(health / "section_grade_guard_latest.json", {"timestamp_utc": now, "overall_status": "ready", "sections": []})
    for name in (
        "runtime_paper_regression_guard_latest.json",
        "stateful_storage_regression_guard_latest.json",
        "one_numbers_regression_guard_latest.json",
    ):
        _write_json(health / name, {"timestamp_utc": now, "ok": True, "overall_status": "ready"})
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "lease_state": "healthy",
            "lease_budget": {"expires_in_seconds": 2400, "critical_lease_seconds": 600},
            "broker_state": {"broker_ready": True, "network_ok": True, "auth_ok": True, "auth_probe_ok": True},
        },
    )
    _write_json(health / "broker_readiness_latest.json", {"timestamp_utc": now, "ready_for_open": True, "auth_ok": True, "network_ok": True})
    _write_json(health / "schwab_auth_supervisor_latest.json", {"timestamp_utc": now, "ok": True, "overall_status": "ready"})
    _write_json(
        health / "source_verification_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "degraded_artifacts": [],
            "stale_artifacts": [],
            "unverified_sources": [],
            "overall": {
                "all_verified": True,
                "total_sources": 6,
                "counts": {"cross_verified": 3, "single_source_verified": 3, "single_source_unverified": 0},
                "unverified_sources": [],
                "low_confidence_sources": [],
                "min_source_confidence_score": 0.82,
            },
        },
    )
    _write_json(
        health / "livefeed_refresh_guard_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "route_checks": [{"name": "feed_refresh_all", "ok": True}],
            "blockers": [],
            "warnings": [],
        },
    )
    _write_json(
        health / "livefeed_local_latest.json",
        {
            "timestamp_utc": now,
            "status": "running",
            "alive": True,
            "health_writer": True,
            "writer_mode": "local_mirror",
            "skipped_file_count": 0,
            "stale_count": 0,
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "recommended_profile": "max_throughput",
            "reasons": ["memory_headroom_ok"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 0.1,
                "compressed_store_gb": 2.0,
            },
            "raw_memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 0.1,
                "compressed_store_gb": 2.0,
            },
            "memory_truth_reconciliation": {"active": False},
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "classification": {"status": "clear"},
            "reopen_gate": {"safe_to_widen_p_core_workers": True, "safe_for_training": True},
            "snapshot": {"memory_truth_reconciliation": {"active": False}},
        },
    )
    _write_json(health / "swap_pressure_governor_latest.json", {"timestamp_utc": now, "ok": True, "overall_status": "ready"})
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "throttle_profile": "soft_cap",
            "compute_pressure_level": "normal",
            "memory_pressure_level": "normal",
            "host_saturation_score": 22.0,
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {
                "total_pending_lines": 0,
                "core_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "storage_plane_contract": {
                "disk_contract": {
                    "emergency_disk_guard": False,
                    "external_available_gb": 90.0,
                    "external_used_percent": 89.0,
                }
            },
        },
    )
    pcore_contract = {
        "active": True,
        "single_writer_only": True,
        "performance_core_primary": True,
        "preprocess_worker_budget": 6,
        "shard_link_writer_lanes": 6,
        "max_shard_link_writer_lanes": 8,
        "p_core_burst_intelligence": {"selected_workers": 6},
        "control_env": {"BACKLOG_PCORE_ALLOCATION_ACTIVE": "1", "SQL_LINK_WRITER_BACKGROUND_POLICY": "0"},
    }
    _write_json(
        health / "backpressure_drainer_fleet_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "handoff_requested",
            "service_request": {"active": True, "p_core_backlog_allocation_contract": pcore_contract},
        },
    )
    _write_json(health / "backlog_pcore_accelerator_latest.json", {"timestamp_utc": now, "ok": False, "overall_status": "advisory"})


def test_adaptive_regression_guard_manages_paper_soak_auth_warning_above_ready_floor(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    health = tmp_path / "governance" / "health"
    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {"expires_in_seconds": 1120, "critical_lease_seconds": 600, "token_lease_grace": True},
            "broker_state": {"broker_ready": True, "broker_operable": True, "network_ok": True, "auth_ok": False, "auth_probe_ok": False},
        },
    )
    _write_json(
        health / "broker_readiness_latest.json",
        {
            "timestamp_utc": now,
            "ready_for_open": True,
            "auth_ok": False,
            "network_ok": True,
            "token_expires_in_seconds": 1120,
        },
    )
    _write_json(
        health / "schwab_auth_supervisor_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "paper_soak_auth_operable": True,
        },
    )

    payload = src.build_payload(
        tmp_path,
        apply=True,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
        feedback_path=tmp_path / "governance" / "health" / "feedback.jsonl",
    )
    broker_auth = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:broker_auth_contract")

    assert payload["overall_status"] == "ready"
    assert broker_auth["state"] == "ready"
    assert broker_auth["adaptive_severity"] == "info"
    assert broker_auth["metrics"]["paper_soak_auth_operable"] is True
    assert broker_auth["metrics"]["blockers"] == []
    assert "broker_auth_not_ok" in broker_auth["metrics"]["managed_auth_blockers"]
    assert "broker_auth_probe_not_ok" in broker_auth["metrics"]["managed_auth_blockers"]


def test_adaptive_regression_guard_marks_persistent_pressure_deferred_storage(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path, high_pressure=True)
    state_path = tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json"
    _write_json(
        state_path,
        {
            "surfaces": {
                "grade:storage_control": {
                    "last_state": "degraded",
                    "consecutive_non_ready_count": 2,
                    "consecutive_blocked_count": 0,
                    "ready_streak": 0,
                    "first_non_ready_utc": "2026-06-21T12:00:00+00:00",
                }
            }
        },
    )

    def guard_builder(_: Path) -> dict:
        return {
            "overall_status": "degraded",
            "blocked_surface_count": 0,
            "degraded_surface_count": 1,
            "surfaces": [
                {
                    "surface": "storage_control",
                    "state": "degraded",
                    "severity": "warning",
                    "summary": "storage recovery is still active",
                    "quiet_hours_preferred": True,
                    "recommended_command": ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
                    "retry_budget": {"step_timeout_sec": 900},
                }
            ],
        }

    payload = src.build_payload(
        tmp_path,
        apply=True,
        persistence_threshold=3,
        grade_guard_builder=guard_builder,
        state_path=state_path,
        feedback_path=tmp_path / "governance" / "health" / "feedback.jsonl",
    )

    storage = next(row for row in payload["surfaces"] if row["surface_id"] == "grade:storage_control")
    written_state = json.loads(state_path.read_text(encoding="utf-8"))

    assert payload["overall_status"] == "degraded"
    assert storage["persistent_regression"] is True
    assert storage["adaptive_action"] == "defer_heavy_repair_until_pressure_cools"
    assert payload["pressure_deferred_count"] == 1
    assert ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"] in payload["recommended_commands"]
    assert written_state["surfaces"]["grade:storage_control"]["consecutive_non_ready_count"] == 3


def test_adaptive_regression_guard_uses_fresher_advisory_runtime_pressure(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path, high_pressure=True)
    health = tmp_path / "governance" / "health"
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    new = datetime.now(timezone.utc).isoformat()
    _write_json(
        health / "health_fast_latest.json",
        {
            "timestamp_utc": old,
            "overall_status": "blocked",
            "runtime_pressure": {
                "overall_status": "blocked",
                "host_saturation_score": 100.0,
                "compute_pressure_level": "high",
                "memory_pressure_level": "elevated",
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": new,
            "overall_status": "advisory",
            "host_saturation_score": 57.0,
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
        },
    )
    state_path = tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json"
    _write_json(
        state_path,
        {
            "surfaces": {
                "grade:storage_control": {
                    "last_state": "degraded",
                    "consecutive_non_ready_count": 2,
                    "consecutive_blocked_count": 0,
                    "ready_streak": 0,
                }
            }
        },
    )

    def guard_builder(_: Path) -> dict:
        return {
            "overall_status": "degraded",
            "blocked_surface_count": 0,
            "degraded_surface_count": 1,
            "surfaces": [
                {
                    "surface": "storage_control",
                    "state": "degraded",
                    "severity": "warning",
                    "summary": "storage recovery is still active",
                    "quiet_hours_preferred": True,
                    "recommended_command": ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
                    "retry_budget": {"step_timeout_sec": 900},
                }
            ],
        }

    payload = src.build_payload(
        tmp_path,
        apply=True,
        persistence_threshold=3,
        grade_guard_builder=guard_builder,
        state_path=state_path,
        feedback_path=tmp_path / "governance" / "health" / "feedback.jsonl",
    )

    storage = next(row for row in payload["surfaces"] if row["surface_id"] == "grade:storage_control")

    assert payload["pressure_context"]["source"] == "runtime_throttle_control"
    assert payload["pressure_context"]["health_fast_ignored_as_stale"] is True
    assert payload["pressure_context"]["high_pressure"] is False
    assert storage["adaptive_action"] == "run_targeted_repair"
    assert payload["pressure_deferred_count"] == 0


def test_adaptive_regression_guard_escalates_repeated_blocked_surface(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    state_path = tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json"
    _write_json(
        state_path,
        {
            "surfaces": {
                "grade:training_lineage": {
                    "last_state": "blocked",
                    "consecutive_non_ready_count": 1,
                    "consecutive_blocked_count": 1,
                    "ready_streak": 0,
                }
            }
        },
    )

    def guard_builder(_: Path) -> dict:
        return {
            "overall_status": "blocked",
            "blocked_surface_count": 1,
            "degraded_surface_count": 0,
            "surfaces": [
                {
                    "surface": "training_lineage",
                    "state": "blocked",
                    "severity": "critical",
                    "summary": "lineage fell below regression floor",
                    "recommended_command": ["./scripts/ops/opsctl.sh", "grade-lift-hardening", "--json"],
                }
            ],
        }

    payload = src.build_payload(
        tmp_path,
        blocked_escalation_threshold=2,
        grade_guard_builder=guard_builder,
        state_path=state_path,
    )

    lineage = next(row for row in payload["surfaces"] if row["surface_id"] == "grade:training_lineage")

    assert payload["overall_status"] == "blocked"
    assert lineage["adaptive_severity"] == "critical"
    assert lineage["adaptive_action"] == "run_guarded_repair"
    assert payload["critical_regression_count"] == 1
    assert ["./scripts/ops/opsctl.sh", "grade-regression-autopilot", "--apply", "--json"] in payload["recommended_commands"]


def test_adaptive_regression_guard_tracks_recovery_ready_streak(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    state_path = tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json"
    _write_json(
        state_path,
        {
            "surfaces": {
                "grade:autonomy_control": {
                    "last_state": "degraded",
                    "consecutive_non_ready_count": 4,
                    "consecutive_blocked_count": 0,
                    "ready_streak": 0,
                }
            }
        },
    )

    def guard_builder(_: Path) -> dict:
        return {
            "overall_status": "ready",
            "blocked_surface_count": 0,
            "degraded_surface_count": 0,
            "surfaces": [
                {
                    "surface": "autonomy_control",
                    "state": "ready",
                    "summary": "autonomy recovered",
                    "recommended_command": ["./scripts/ops/opsctl.sh", "autonomy-control", "--json"],
                }
            ],
        }

    payload = src.build_payload(
        tmp_path,
        apply=True,
        grade_guard_builder=guard_builder,
        state_path=state_path,
        feedback_path=tmp_path / "governance" / "health" / "feedback.jsonl",
    )

    autonomy = next(row for row in payload["surfaces"] if row["surface_id"] == "grade:autonomy_control")
    written_state = json.loads(state_path.read_text(encoding="utf-8"))

    assert payload["overall_status"] == "ready"
    assert payload["recovered_surface_count"] == 1
    assert autonomy["adaptive_action"] == "watch_recovery"
    assert written_state["surfaces"]["grade:autonomy_control"]["ready_streak"] == 1


def test_adaptive_regression_guard_includes_critical_contract_surfaces(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    rows = {row["surface_id"]: row for row in payload["surfaces"]}

    for surface_id in src.CRITICAL_CONTRACT_IDS:
        assert rows[surface_id]["state"] == "ready"
        assert rows[surface_id]["critical_contract"] is True
    assert payload["overall_status"] == "ready"


def test_adaptive_regression_guard_blocks_unreconciled_stale_memory_high_water(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "recommended_profile": "air_safe",
            "reasons": ["compressed_memory_high"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 0.2,
                "compressed_store_gb": 7.5,
            },
            "raw_memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 8.2,
                "compressed_store_gb": 22.0,
            },
            "memory_truth_reconciliation": {"active": False},
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "classification": {"status": "clear"},
            "reopen_gate": {"safe_to_widen_p_core_workers": False, "safe_for_training": False},
            "snapshot": {"memory_truth_reconciliation": {"active": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    memory = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:memory_truth_contract")

    assert payload["overall_status"] == "blocked"
    assert memory["state"] == "blocked"
    assert memory["adaptive_severity"] == "critical"
    assert "stale_high_water_memory_not_reconciled" in memory["metrics"]["blockers"]
    assert "green_memory_has_high_pressure_reason" in memory["metrics"]["blockers"]
    assert ["./scripts/ops/opsctl.sh", "memory-efficiency", "apply", "--json"] in payload["recommended_commands"]


def test_adaptive_regression_guard_softens_training_debt_during_guarded_paper(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "section_grade_guard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "ok": True,
            "guarded_paper_ready": True,
            "live_execution_locked": True,
            "paper_soak_advisory_below_floor": True,
            "advisory_below_floor_sections": ["training_and_model_quality"],
            "sections": [
                {
                    "section": "training_and_model_quality",
                    "state": "below_floor",
                    "score": 87.75,
                    "raw_score": 87.75,
                    "letter_grade": "B+",
                    "raw_letter_grade": "B+",
                    "floor_reason": "training quality debt needs more collection",
                    "recommended_commands": [["./scripts/ops/opsctl.sh", "training-quality", "--json"]],
                }
            ],
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {
            "overall_status": "blocked",
            "blocked_surface_count": 1,
            "degraded_surface_count": 0,
            "surfaces": [
                {
                    "surface": "training_quality",
                    "state": "blocked",
                    "severity": "critical",
                    "summary": "training_quality_score=56.00 regressed below the safe floor",
                    "metrics": {"training_quality_score": 56.0},
                    "recommended_command": ["./scripts/ops/opsctl.sh", "training-quality", "--json"],
                }
            ],
        },
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    rows = {row["surface_id"]: row for row in payload["surfaces"]}

    assert payload["overall_status"] == "ready"
    assert payload["active_regression_count"] == 0
    assert rows["grade:training_quality"]["state"] == "ready"
    assert rows["grade:training_quality"]["metrics"]["paper_soak_advisory"] is True
    assert rows["grade:training_quality"]["metrics"]["paper_soak_quality_advisory_only"] is True
    assert rows["grade:training_quality"]["metrics"]["original_state"] == "blocked"
    assert rows["section:training_and_model_quality"]["state"] == "ready"
    assert rows["section:training_and_model_quality"]["metrics"]["paper_soak_advisory"] is True
    assert rows["section:training_and_model_quality"]["metrics"]["paper_soak_quality_advisory_only"] is True


def test_adaptive_regression_guard_softens_paper_soak_section_advisories(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"timestamp_utc": now, "overall_status": "ready", "clearance_plan": {"clearance_state": "managed_coverage_stage_deferred"}},
    )
    _write_json(
        health / "section_grade_guard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "ok": True,
            "guarded_paper_ready": True,
            "live_execution_locked": True,
            "paper_soak_advisory_below_floor": True,
            "advisory_below_floor_sections": [
                "data_ingestion_and_storage",
                "live_trading_readiness",
                "ops_and_autonomy",
            ],
            "sections": [
                {
                    "section": "data_ingestion_and_storage",
                    "state": "below_floor",
                    "score": 84.53,
                    "raw_score": 84.53,
                    "letter_grade": "B+",
                    "raw_letter_grade": "B+",
                    "floor_reason": "bounded storage recovery remains active",
                    "recommended_commands": [["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]],
                },
                {
                    "section": "live_trading_readiness",
                    "state": "below_floor",
                    "score": 86.0,
                    "raw_score": 86.0,
                    "letter_grade": "B+",
                    "raw_letter_grade": "B+",
                    "recommended_commands": [["./scripts/ops/opsctl.sh", "live-canary-control", "--json"]],
                },
                {
                    "section": "ops_and_autonomy",
                    "state": "below_floor",
                    "score": 86.0,
                    "raw_score": 86.0,
                    "letter_grade": "B+",
                    "raw_letter_grade": "B+",
                    "recommended_commands": [["./scripts/ops/opsctl.sh", "autonomy-control", "--json"]],
                },
            ],
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    rows = {row["surface_id"]: row for row in payload["surfaces"]}

    assert payload["overall_status"] == "ready"
    assert payload["active_regression_count"] == 0
    assert rows["section:data_ingestion_and_storage"]["state"] == "ready"
    assert rows["section:live_trading_readiness"]["state"] == "ready"
    assert rows["section:ops_and_autonomy"]["state"] == "ready"
    assert rows["section:data_ingestion_and_storage"]["metrics"]["paper_soak_section_advisory_only"] is True
    assert rows["section:live_trading_readiness"]["metrics"]["paper_soak_section_advisory_only"] is True
    assert rows["section:ops_and_autonomy"]["metrics"]["paper_soak_section_advisory_only"] is True


def test_adaptive_regression_guard_softens_promotion_gates_during_guarded_paper(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {
            "overall_status": "degraded",
            "blocked_surface_count": 0,
            "degraded_surface_count": 4,
            "surfaces": [
                {
                    "surface": "training_lineage",
                    "state": "degraded",
                    "severity": "warning",
                    "summary": "lineage_score=92.50 with seeded recovery evidence still needs final replay and signing proof",
                    "metrics": {"lineage_score": 92.5},
                    "recommended_command": ["./scripts/ops/opsctl.sh", "grade-lift-hardening", "--json"],
                },
                {
                    "surface": "live_canary",
                    "state": "degraded",
                    "severity": "warning",
                    "summary": "recommended_mode=preapproved_supervised is still staged, not supervised",
                    "metrics": {"recommended_mode": "preapproved_supervised"},
                    "recommended_command": ["./scripts/ops/opsctl.sh", "live-canary-control", "--json"],
                },
                {
                    "surface": "autonomy_control",
                    "state": "degraded",
                    "severity": "warning",
                    "summary": "autonomy_score=98.52 is stable enough to protect gains but not yet self-clearing",
                    "metrics": {"autonomy_score": 98.52},
                    "recommended_command": ["./scripts/ops/opsctl.sh", "autonomy-control", "--json"],
                },
                {
                    "surface": "promotion_autopilot",
                    "state": "degraded",
                    "severity": "warning",
                    "summary": "packet_completeness_score=75.00 with repairable gates still open",
                    "metrics": {"packet_completeness_score": 75.0},
                    "recommended_command": ["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"],
                },
            ],
        },
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    rows = {row["surface_id"]: row for row in payload["surfaces"]}

    assert payload["overall_status"] == "ready"
    assert payload["active_regression_count"] == 0
    for surface_id in src.PAPER_SOAK_PROMOTION_GATE_SURFACES:
        assert rows[surface_id]["state"] == "ready"
        assert rows[surface_id]["adaptive_severity"] == "info"
        assert rows[surface_id]["metrics"]["paper_soak_promotion_gate_advisory_only"] is True
        assert rows[surface_id]["metrics"]["does_not_block_guarded_paper_soak"] is True


def test_adaptive_regression_guard_softens_current_guarded_paper_operational_debt(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {
            "overall_status": "blocked",
            "blocked_surface_count": 3,
            "degraded_surface_count": 0,
            "surfaces": [
                {
                    "surface": "training_quality",
                    "state": "blocked",
                    "severity": "critical",
                    "summary": "training_quality_score=38.88 regressed below the safe floor",
                    "metrics": {"training_quality_score": 38.88},
                    "recommended_command": ["./scripts/ops/opsctl.sh", "training-quality", "--json"],
                },
                {
                    "surface": "incident_closeout",
                    "state": "blocked",
                    "severity": "critical",
                    "summary": "open_incident_count=2 remains a regression blocker",
                    "metrics": {"open_incident_count": 2},
                    "recommended_command": ["./scripts/ops/opsctl.sh", "incident-closeout", "--json"],
                },
                {
                    "surface": "live_canary",
                    "state": "blocked",
                    "severity": "critical",
                    "summary": "live canary fell below staged preclearance",
                    "metrics": {"recommended_mode": "validate_only"},
                    "recommended_command": ["./scripts/ops/opsctl.sh", "live-canary-control", "--json"],
                },
            ],
        },
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    rows = {row["surface_id"]: row for row in payload["surfaces"]}

    assert payload["overall_status"] == "degraded"
    assert payload["critical_regression_count"] == 0
    assert rows["grade:training_quality"]["state"] == "ready"
    assert rows["grade:training_quality"]["metrics"]["paper_soak_advisory"] is True
    assert rows["grade:training_quality"]["metrics"]["paper_soak_quality_advisory_only"] is True
    assert rows["grade:incident_closeout"]["state"] == "degraded"
    assert rows["grade:incident_closeout"]["metrics"]["health_fast_strict_clear"] is True
    assert rows["grade:live_canary"]["state"] == "ready"
    assert rows["grade:live_canary"]["metrics"]["paper_soak_promotion_gate_advisory_only"] is True
    assert rows["grade:live_canary"]["metrics"]["original_state"] == "blocked"


def test_adaptive_regression_guard_keeps_promotion_gate_degraded_without_guarded_paper_lock(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {
            "overall_status": "degraded",
            "blocked_surface_count": 0,
            "degraded_surface_count": 1,
            "surfaces": [
                {
                    "surface": "promotion_autopilot",
                    "state": "degraded",
                    "severity": "warning",
                    "summary": "packet_completeness_score=75.00 with repairable gates still open",
                    "metrics": {"packet_completeness_score": 75.0},
                    "recommended_command": ["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"],
                }
            ],
        },
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    promotion = next(row for row in payload["surfaces"] if row["surface_id"] == "grade:promotion_autopilot")

    assert payload["overall_status"] == "degraded"
    assert promotion["state"] == "degraded"
    assert promotion["metrics"].get("paper_soak_promotion_gate_advisory_only") is None


def test_adaptive_regression_guard_softens_optional_source_debt_during_guarded_paper(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    optional_sources = sorted(src.OPTIONAL_CONTEXT_SOURCE_IDS)
    _write_json(
        health / "source_verification_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "degraded",
            "degraded_artifacts": optional_sources,
            "stale_artifacts": optional_sources,
            "unverified_sources": optional_sources,
            "overall": {
                "all_verified": False,
                "total_sources": 16,
                "counts": {"cross_verified": 2, "single_source_verified": 6, "single_source_unverified": len(optional_sources)},
                "unverified_sources": optional_sources,
                "low_confidence_sources": optional_sources,
                "min_source_confidence_score": 0.22,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    source = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:source_verification_contract")

    assert payload["overall_status"] == "ready"
    assert source["state"] == "ready"
    assert source["adaptive_severity"] == "info"
    assert source["metrics"]["optional_context_source_debt"] is True
    assert source["metrics"]["optional_context_advisory_only"] is True
    assert source["metrics"]["blockers"] == []
    assert "source_verification_optional_context_debt_for_guarded_paper" in source["metrics"]["warnings"]


def test_adaptive_regression_guard_does_not_mix_verified_warning_with_optional_unverified_debt(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    optional_sources = ["public_macro_feeds", "sec_edgar_context"]
    _write_json(
        health / "source_verification_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "degraded",
            "degraded_artifacts": [*optional_sources, "market_quote_profiles"],
            "stale_artifacts": optional_sources,
            "unverified_sources": optional_sources,
            "sources": [
                {"source_id": "market_quote_profiles", "verification_status": "cross_verified"},
                {"source_id": "public_macro_feeds", "verification_status": "single_source_unverified"},
                {"source_id": "sec_edgar_context", "verification_status": "single_source_unverified"},
            ],
            "overall": {
                "all_verified": False,
                "total_sources": 16,
                "counts": {"cross_verified": 3, "single_source_verified": 9, "single_source_unverified": 2},
                "unverified_sources": optional_sources,
                "low_confidence_sources": optional_sources,
                "min_source_confidence_score": 0.42,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    source = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:source_verification_contract")

    assert payload["overall_status"] == "ready"
    assert source["state"] == "ready"
    assert source["metrics"]["optional_context_source_debt"] is True
    assert source["metrics"]["verified_warning_artifact_count"] == 1
    assert source["metrics"]["unverified_degraded_artifact_count"] == 2
    assert source["metrics"]["blockers"] == []


def test_adaptive_regression_guard_softens_optional_source_debt_when_live_is_read_only(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "degraded",
            "read_only": True,
            "operational_readiness": {
                "guarded_paper": {"ok": False, "status": "blocked", "blockers": ["global_clear_blocker=auth_lease_critical"]},
                "live_execution": {
                    "ok": False,
                    "status": "blocked_read_only",
                    "blockers": ["live_execution_requires_explicit_operator_control"],
                },
            },
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "degraded",
            "degraded_artifacts": ["ticker_news_context"],
            "stale_artifacts": ["ticker_news_context"],
            "unverified_sources": ["ticker_news_context"],
            "overall": {
                "all_verified": False,
                "total_sources": 16,
                "counts": {"cross_verified": 3, "single_source_verified": 12, "single_source_unverified": 1},
                "unverified_sources": ["ticker_news_context"],
                "low_confidence_sources": [],
                "min_source_confidence_score": 0.41,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    source = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:source_verification_contract")

    assert payload["overall_status"] == "ready"
    assert source["state"] == "ready"
    assert source["metrics"]["optional_context_source_debt"] is True
    assert source["metrics"]["optional_context_advisory_only"] is True
    assert source["metrics"]["read_only_source_advisory_context"] is True
    assert source["metrics"]["blockers"] == []


def test_adaptive_regression_guard_softens_current_read_only_context_source_debt(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    source_ids = ["crypto_market_context", "public_macro_feeds", "market_micro_context"]
    _write_json(
        health / "source_verification_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "degraded",
            "degraded_artifacts": source_ids,
            "stale_artifacts": source_ids,
            "unverified_sources": source_ids,
            "overall": {
                "all_verified": False,
                "total_sources": 16,
                "counts": {"cross_verified": 2, "single_source_verified": 6, "single_source_unverified": len(source_ids)},
                "unverified_sources": source_ids,
                "low_confidence_sources": source_ids,
                "min_source_confidence_score": 0.45,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    source = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:source_verification_contract")

    assert payload["overall_status"] == "ready"
    assert source["state"] == "ready"
    assert source["metrics"]["optional_context_source_debt"] is True
    assert source["metrics"]["optional_context_advisory_only"] is True
    assert source["metrics"]["blockers"] == []


def test_adaptive_regression_guard_keeps_required_source_debt_blocking(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "source_verification_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "degraded",
            "degraded_artifacts": ["broker_execution_truth"],
            "stale_artifacts": [],
            "unverified_sources": ["broker_execution_truth"],
            "overall": {
                "all_verified": False,
                "total_sources": 12,
                "counts": {"cross_verified": 3, "single_source_verified": 8, "single_source_unverified": 1},
                "unverified_sources": ["broker_execution_truth"],
                "low_confidence_sources": [],
                "min_source_confidence_score": 0.82,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    source = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:source_verification_contract")

    assert payload["overall_status"] == "blocked"
    assert source["state"] == "blocked"
    assert source["adaptive_severity"] == "critical"
    assert source["metrics"]["optional_context_source_debt"] is False
    assert "source_verification_unverified_sources" in source["metrics"]["blockers"]


def test_adaptive_regression_guard_softens_memory_soft_guard_during_guarded_paper(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "needs_work",
            "recommended_profile": "air_safe",
            "reasons": ["compressed_memory_high"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 1.9,
                "compressed_store_gb": 17.8,
            },
            "raw_memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 1.9,
                "compressed_store_gb": 17.8,
            },
            "memory_truth_reconciliation": {"active": False},
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "advisory",
            "classification": {"status": "foreground_headroom"},
            "reopen_gate": {"safe_to_widen_p_core_workers": False, "safe_for_training": False},
            "snapshot": {"pages_throttled": 0.0, "memory_truth_reconciliation": {"active": False}},
        },
    )
    _write_json(health / "swap_pressure_governor_latest.json", {"timestamp_utc": now, "ok": True, "overall_status": "ready"})

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    memory = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:memory_truth_contract")

    assert payload["overall_status"] == "ready"
    assert memory["state"] == "ready"
    assert memory["adaptive_severity"] == "info"
    assert memory["metrics"]["memory_soft_guard_for_paper_soak"] is True
    assert memory["metrics"]["paper_soak_soft_guard_advisory_only"] is True
    assert memory["metrics"]["safe_for_training"] is False
    assert memory["metrics"]["blockers"] == []
    assert "memory_soft_guard_for_guarded_paper" in memory["metrics"]["warnings"]


def test_adaptive_regression_guard_accepts_benign_memory_efficiency_advisory_with_normal_swap(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "advisory",
            "recommended_profile": "air_safe",
            "reasons": ["creative_session_music_playback"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "memory_free_pct": 90.0,
                "swap_used_gb": 4.008,
                "compressed_store_gb": 8.71,
            },
            "raw_memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "memory_free_pct": 90.0,
                "swap_used_gb": 4.008,
                "compressed_store_gb": 8.71,
            },
            "memory_truth_reconciliation": {"active": False},
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "classification": {"status": "clear"},
            "reopen_gate": {"safe_to_widen_p_core_workers": True, "safe_for_training": True},
            "snapshot": {"pages_throttled": 0.0, "memory_truth_reconciliation": {"active": False}},
        },
    )
    _write_json(
        health / "swap_pressure_governor_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "swap_pressure": {
                "tier": "normal",
                "swap_used_gb": 4.008,
                "thresholds": {"calm_swap_gb": 10.0},
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    memory = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:memory_truth_contract")

    assert payload["overall_status"] == "ready"
    assert memory["state"] == "ready"
    assert memory["summary"].startswith("efficiency=advisory_ready")
    assert memory["metrics"]["memory_efficiency_advisory_ready"] is True
    assert memory["metrics"]["swap_pressure_tier"] == "normal"
    assert memory["metrics"]["calm_swap_limit_gb"] == 10.0
    assert memory["metrics"]["blockers"] == []
    assert "memory_efficiency_advisory_ready" in memory["metrics"]["warnings"]

    constrained_swap = json.loads((health / "swap_pressure_governor_latest.json").read_text(encoding="utf-8"))
    constrained_swap["swap_pressure"]["tier"] = "constrained"
    _write_json(health / "swap_pressure_governor_latest.json", constrained_swap)
    blocked = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state_constrained.json",
    )
    blocked_memory = next(row for row in blocked["surfaces"] if row["surface_id"] == "guard:memory_truth_contract")

    assert blocked["overall_status"] == "blocked"
    assert blocked_memory["metrics"]["memory_efficiency_advisory_ready"] is False
    assert "memory_efficiency_not_ready" in blocked_memory["metrics"]["blockers"]


def test_adaptive_regression_guard_treats_stale_swap_governor_as_advisory_when_memory_is_green(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc)
    stale = (now - timedelta(minutes=45)).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "ready",
            "recommended_profile": "max_throughput",
            "reasons": ["memory_headroom_ok"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 1.5,
                "compressed_store_gb": 5.2,
            },
            "raw_memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 1.5,
                "compressed_store_gb": 5.2,
            },
            "memory_truth_reconciliation": {"active": False},
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "ready",
            "classification": {"status": "clear"},
            "reopen_gate": {"safe_to_widen_p_core_workers": True, "safe_for_training": True},
            "snapshot": {"pages_throttled": 0.0, "memory_truth_reconciliation": {"active": False}},
        },
    )
    _write_json(
        health / "swap_pressure_governor_latest.json",
        {
            "timestamp_utc": stale,
            "ok": True,
            "overall_status": "ready",
            "swap_pressure": {"tier": "normal", "swap_used_gb": 1.5},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    memory = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:memory_truth_contract")

    assert payload["overall_status"] == "ready"
    assert memory["state"] == "ready"
    assert memory["metrics"]["paper_soak_soft_guard_advisory_only"] is True
    assert "swap_pressure_governor_stale" in memory["metrics"]["warnings"]
    assert "memory_soft_guard_for_guarded_paper" in memory["metrics"]["warnings"]


def test_adaptive_regression_guard_treats_foreground_memory_advisory_as_ready(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "advisory",
            "classification": {"status": "foreground_headroom"},
            "reopen_gate": {
                "safe_to_widen_p_core_workers": True,
                "safe_for_training": False,
                "small_batch_training_safe": True,
            },
            "snapshot": {"pages_throttled": 0.0, "memory_truth_reconciliation": {"active": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    memory = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:memory_truth_contract")

    assert payload["overall_status"] == "ready"
    assert memory["state"] == "ready"
    assert memory["metrics"]["memory_pressure_advisory_ready"] is True
    assert memory["metrics"]["safe_to_widen_p_core_workers"] is True
    assert memory["metrics"]["safe_for_training"] is False
    assert memory["metrics"]["blockers"] == []
    assert "memory_pressure_advisory_ready" in memory["metrics"]["warnings"]


def test_adaptive_regression_guard_softens_unknown_green_memory_advisory_during_guarded_paper(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "advisory",
            "classification": {"status": "vm_green_new_advisory_label"},
            "reopen_gate": {"safe_to_widen_p_core_workers": False, "safe_for_training": False},
            "snapshot": {"pages_throttled": 0.0, "memory_truth_reconciliation": {"active": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    memory = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:memory_truth_contract")

    assert payload["overall_status"] == "ready"
    assert memory["state"] == "ready"
    assert memory["metrics"]["memory_soft_guard_for_paper_soak"] is True
    assert memory["metrics"]["classification_soft_or_clear"] is True
    assert memory["metrics"]["blockers"] == []


def test_adaptive_regression_guard_treats_slightly_stale_green_memory_as_soft_soak_guard(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    _write_guarded_paper_health_fast(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    stale = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "timestamp_utc": stale,
            "ok": True,
            "overall_status": "ready",
            "recommended_profile": "max_throughput",
            "reasons": ["memory_headroom_ok"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 0.4,
                "compressed_store_gb": 4.5,
            },
            "raw_memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 0.4,
                "compressed_store_gb": 4.5,
            },
            "memory_truth_reconciliation": {"active": False},
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "classification": {"status": "clear"},
            "reopen_gate": {"safe_to_widen_p_core_workers": True, "safe_for_training": True},
            "snapshot": {"pages_throttled": 0.0, "memory_truth_reconciliation": {"active": False}},
        },
    )
    _write_json(health / "swap_pressure_governor_latest.json", {"timestamp_utc": now, "ok": True, "overall_status": "ready"})

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    memory = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:memory_truth_contract")

    assert payload["overall_status"] == "ready"
    assert memory["state"] == "ready"
    assert memory["metrics"]["paper_soak_soft_guard_advisory_only"] is True
    assert "memory_efficiency_stale" in memory["metrics"]["warnings"]


def test_adaptive_regression_guard_treats_runtime_advisory_with_clear_storage_as_ready(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "advisory",
            "throttle_profile": "soft_cap",
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
            "host_saturation_score": 45.65,
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.009,
            "backpressure": {
                "total_pending_lines": 2634,
                "core_pending_lines": 71,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    runtime_storage = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:runtime_storage_contract")

    assert payload["overall_status"] == "ready"
    assert runtime_storage["state"] == "ready"
    assert runtime_storage["adaptive_severity"] == "info"
    assert runtime_storage["metrics"]["runtime_advisory_ready"] is True
    assert runtime_storage["metrics"]["storage_clear"] is True
    assert runtime_storage["metrics"]["blockers"] == []
    assert runtime_storage["metrics"]["warnings"] == []


def test_adaptive_regression_guard_treats_managed_high_compute_advisory_as_ready(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "advisory",
            "throttle_profile": "sustain",
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "host_saturation_score": 51.86,
            "soft_cap_advisory_reclassification": {
                "active": True,
                "reason": "research_training_pressure_is_already_niced_and_guarded_advisory",
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.009,
            "backpressure": {
                "total_pending_lines": 2634,
                "core_pending_lines": 71,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    runtime_storage = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:runtime_storage_contract")

    assert payload["overall_status"] == "ready"
    assert runtime_storage["state"] == "ready"
    assert runtime_storage["metrics"]["runtime_advisory_ready"] is True
    assert runtime_storage["metrics"]["managed_high_compute_advisory"] is True
    assert runtime_storage["metrics"]["blockers"] == []


def test_adaptive_regression_guard_treats_external_dominant_high_compute_as_soak_advisory(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "blocked",
            "throttle_profile": "protect_live",
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "host_saturation_score": 82.0,
            "host_pressure_attribution": {
                "external_pressure_dominant": True,
                "bot_owned_pressure_dominant": False,
            },
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {
                "total_pending_lines": 0,
                "core_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    runtime_storage = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:runtime_storage_contract")

    assert payload["overall_status"] == "degraded"
    assert payload["critical_regression_count"] == 0
    assert runtime_storage["state"] == "degraded"
    assert runtime_storage["metrics"]["runtime_advisory_ready"] is True
    assert runtime_storage["metrics"]["external_pressure_advisory"] is True
    assert runtime_storage["metrics"]["managed_high_compute_advisory"] is True
    assert runtime_storage["metrics"]["blockers"] == []
    assert "host_saturation_elevated" in runtime_storage["metrics"]["warnings"]


def test_adaptive_regression_guard_accepts_capacity_limited_paper_runtime(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "degraded",
            "throttle_profile": "soft_cap",
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
            "host_saturation_score": 49.5,
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "capacity_limited_paper_execution": True,
                "reason": "paper_ramp_armed_capacity_limited_for_full_force_soak",
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {
                "total_pending_lines": 0,
                "core_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    runtime_storage = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:runtime_storage_contract")

    assert payload["overall_status"] == "ready"
    assert runtime_storage["state"] == "ready"
    assert runtime_storage["metrics"]["runtime_advisory_ready"] is True
    assert runtime_storage["metrics"]["capacity_limited_paper_advisory"] is True
    assert runtime_storage["metrics"]["blockers"] == []
    assert runtime_storage["metrics"]["warnings"] == []


def test_adaptive_regression_guard_accepts_runtime_verified_guarded_saturation(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "throttle_profile": "sustain",
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
            "host_saturation_score": 63.9,
            "soft_cap_advisory_reclassification": {
                "active": True,
                "to_status": "ready",
                "measurements": {"runtime_ready_guarded": True},
            },
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {
                "total_pending_lines": 0,
                "core_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {
            "overall_status": "ready",
            "blocked_surface_count": 0,
            "degraded_surface_count": 0,
            "surfaces": [],
        },
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    runtime_storage = next(
        row for row in payload["surfaces"] if row["surface_id"] == "guard:runtime_storage_contract"
    )

    assert payload["overall_status"] == "ready"
    assert runtime_storage["state"] == "ready"
    assert runtime_storage["metrics"]["guarded_runtime_ready_saturation"] is True
    assert runtime_storage["metrics"]["warnings"] == []


def test_adaptive_regression_guard_accepts_normal_compute_armed_paper_capacity(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "degraded",
            "throttle_profile": "soft_cap",
            "compute_pressure_level": "normal",
            "memory_pressure_level": "normal",
            "host_saturation_score": 49.5,
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "reason": "paper_ramp_armed_and_clean",
                "stage": "armed",
                "armed": True,
                "ok": True,
                "blockers": [],
            },
        },
    )
    _write_json(
        health / "paper_400_ramp_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "stage": "armed",
            "armed": True,
            "blockers": [],
            "gates": {
                "runtime": {
                    "ok": True,
                    "status": "ready",
                    "blockers": [],
                    "runtime_capacity_ready": True,
                    "capacity_limited_armed": True,
                    "paper_execution_clean": True,
                    "live_execution_locked": True,
                    "pressure_limited": False,
                    "compute_pressure_level": "normal",
                    "memory_pressure_level": "normal",
                }
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {
                "total_pending_lines": 0,
                "core_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    runtime_storage = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:runtime_storage_contract")

    assert payload["overall_status"] == "ready"
    assert runtime_storage["state"] == "ready"
    assert runtime_storage["metrics"]["runtime_advisory_ready"] is True
    assert runtime_storage["metrics"]["armed_paper_capacity_advisory"] is True
    assert runtime_storage["metrics"]["capacity_limited_paper_advisory"] is True
    assert runtime_storage["metrics"]["warnings"] == []


def test_adaptive_regression_guard_accepts_capacity_limited_armed_paper_ramp_under_high_compute(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "throttle_profile": "sustain",
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "host_saturation_score": 66.64,
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "reason": "paper_ramp_armed_and_clean",
                "stage": "armed",
                "armed": True,
                "ok": True,
                "blockers": [],
            },
            "runtime_saturation_governor_v2": {
                "paper_live_data_policy": {
                    "paper_execution_allowed": True,
                    "paper_execution_consumer_paused": False,
                    "protect_live_execution_read_only": True,
                    "protect_paper_execution_queue": True,
                }
            },
        },
    )
    _write_json(
        health / "paper_400_ramp_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "stage": "armed",
            "armed": True,
            "blockers": [],
            "gates": {
                "runtime": {
                    "ok": True,
                    "status": "capacity_limited_armed",
                    "blockers": [],
                    "runtime_capacity_ready": True,
                    "capacity_limited_armed": True,
                    "paper_execution_clean": True,
                    "live_execution_locked": True,
                    "pressure_limited": True,
                    "compute_pressure_level": "high",
                    "memory_pressure_level": "normal",
                }
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {
                "total_pending_lines": 0,
                "core_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    runtime_storage = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:runtime_storage_contract")

    assert payload["overall_status"] == "ready"
    assert runtime_storage["state"] == "ready"
    assert runtime_storage["metrics"]["runtime_advisory_ready"] is True
    assert runtime_storage["metrics"]["capacity_limited_paper_advisory"] is True
    assert runtime_storage["metrics"]["paper_capacity_limited_armed"] is True
    assert runtime_storage["metrics"]["paper_execution_open"] is True
    assert runtime_storage["metrics"]["blockers"] == []
    assert runtime_storage["metrics"]["warnings"] == []


def test_adaptive_regression_guard_blocks_inactive_pcore_backlog_contract(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "backpressure_drainer_fleet_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "handoff_requested",
            "service_request": {
                "active": True,
                "p_core_backlog_allocation_contract": {
                    "active": False,
                    "single_writer_only": False,
                    "performance_core_primary": False,
                    "preprocess_worker_budget": 2,
                    "shard_link_writer_lanes": 9,
                    "max_shard_link_writer_lanes": 6,
                    "p_core_burst_intelligence": {"selected_workers": 2},
                    "control_env": {"BACKLOG_PCORE_ALLOCATION_ACTIVE": "0", "SQL_LINK_WRITER_BACKGROUND_POLICY": "1"},
                },
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    pcore = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:backlog_pcore_contract")

    assert payload["overall_status"] == "blocked"
    assert pcore["state"] == "blocked"
    assert pcore["adaptive_severity"] == "critical"
    assert "p_core_allocation_not_active" in pcore["metrics"]["blockers"]
    assert "p_core_workers_below_floor" in pcore["metrics"]["blockers"]
    assert ["./scripts/ops/opsctl.sh", "backlog-pcore-accelerator", "--apply", "--json"] in payload["recommended_commands"]


def test_adaptive_regression_guard_treats_stale_pcore_support_artifacts_as_advisory_when_contract_is_clear(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    health = tmp_path / "governance" / "health"
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    contract = {
        "active": True,
        "single_writer_only": True,
        "performance_core_primary": True,
        "preprocess_worker_budget": 7,
        "shard_link_writer_lanes": 7,
        "max_shard_link_writer_lanes": 8,
        "p_core_burst_intelligence": {"selected_workers": 7},
        "control_env": {"BACKLOG_PCORE_ALLOCATION_ACTIVE": "1", "SQL_LINK_WRITER_BACKGROUND_POLICY": "0"},
    }
    _write_json(
        health / "backpressure_drainer_fleet_latest.json",
        {
            "timestamp_utc": stale,
            "ok": True,
            "overall_status": "handoff_requested",
            "service_request": {"active": True, "p_core_backlog_allocation_contract": contract},
        },
    )
    _write_json(
        health / "backlog_pcore_accelerator_latest.json",
        {"timestamp_utc": stale, "ok": True, "overall_status": "ready"},
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    pcore = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:backlog_pcore_contract")

    assert payload["overall_status"] == "ready"
    assert pcore["state"] == "ready"
    assert pcore["metrics"]["pcore_contract_operationally_clear"] is True
    assert pcore["metrics"]["stale_support_advisory_only"] is True
    assert pcore["metrics"]["blockers"] == []


def test_adaptive_regression_guard_accepts_five_pcore_workers_when_backlog_is_green(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    health = tmp_path / "governance" / "health"
    now = datetime.now(timezone.utc).isoformat()
    contract = {
        "active": True,
        "single_writer_only": True,
        "performance_core_primary": True,
        "preprocess_worker_budget": 5,
        "shard_link_writer_lanes": 5,
        "max_shard_link_writer_lanes": 8,
        "p_core_burst_intelligence": {"selected_workers": 5},
        "control_env": {"BACKLOG_PCORE_ALLOCATION_ACTIVE": "1", "SQL_LINK_WRITER_BACKGROUND_POLICY": "0"},
    }
    _write_json(
        health / "backpressure_drainer_fleet_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "service_request": {"active": True, "p_core_backlog_allocation_contract": contract},
        },
    )
    _write_json(
        health / "backlog_pcore_accelerator_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "advisory",
            "storage_contract": {
                "green": True,
                "line_green": True,
                "age_green": True,
                "overlay_green": True,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    pcore = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:backlog_pcore_contract")

    assert pcore["state"] == "ready"
    assert pcore["metrics"]["backlog_green"] is True
    assert pcore["metrics"]["active_catchup_target_required"] is False
    assert pcore["metrics"]["operational_worker_floor"] == 4
    assert pcore["metrics"]["pcore_contract_operationally_clear"] is True


def test_adaptive_regression_guard_honors_green_backlog_memory_worker_cap(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    health = tmp_path / "governance" / "health"
    now = datetime.now(timezone.utc).isoformat()
    contract = {
        "active": True,
        "single_writer_only": True,
        "performance_core_primary": True,
        "preprocess_worker_budget": 3,
        "shard_link_writer_lanes": 3,
        "max_shard_link_writer_lanes": 8,
        "p_core_burst_intelligence": {"selected_workers": 3},
        "control_env": {"BACKLOG_PCORE_ALLOCATION_ACTIVE": "1", "SQL_LINK_WRITER_BACKGROUND_POLICY": "0"},
    }
    _write_json(
        health / "backpressure_drainer_fleet_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "service_request": {"active": True, "p_core_backlog_allocation_contract": contract},
        },
    )
    _write_json(
        health / "backlog_pcore_accelerator_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "advisory",
            "host_lane_contract": {"memory_worker_cap": 3},
            "storage_contract": {"green": True, "line_green": True, "age_green": True, "overlay_green": True},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    pcore = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:backlog_pcore_contract")

    assert pcore["state"] == "ready"
    assert pcore["metrics"]["operational_worker_floor"] == 3
    assert pcore["metrics"]["memory_worker_cap"] == 3
    assert pcore["metrics"]["blockers"] == []


def test_adaptive_regression_guard_blocks_ingestion_storage_past_floor(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": False,
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 3.4,
            "recovery_quality_score": 70.0,
            "backpressure_quality_score": 42.0,
            "backpressure": {
                "total_pending_lines": 92000,
                "core_pending_lines": 64000,
                "deferred_pending_lines": 21000,
                "support_pending_lines": 7000,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 3200.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "collector_intake_enforcement_audit": {
                "status": "partial",
                "required": True,
                "mismatch_count": 2,
                "mismatches": [{"key": "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG"}],
            },
            "backlog_relief_contract": {
                "active": False,
                "p_core_backlog_allocation_contract": {
                    "active": False,
                    "single_writer_only": False,
                    "preprocess_worker_budget": 2,
                    "p_core_burst_intelligence": {"selected_workers": 2},
                    "control_env": {"BACKLOG_PCORE_ALLOCATION_ACTIVE": "0"},
                },
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    floor = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:ingestion_storage_degradation_floor")

    assert payload["overall_status"] == "blocked"
    assert floor["state"] == "blocked"
    assert floor["adaptive_severity"] == "critical"
    assert "storage_pressure_index_beyond_floor" in floor["metrics"]["blockers"]
    assert "storage_total_pending_beyond_floor" in floor["metrics"]["blockers"]
    assert "collector_intake_not_enforced_during_pressure" in floor["metrics"]["blockers"]
    assert "p_core_contract_not_active_during_hard_pressure" in floor["metrics"]["blockers"]
    assert ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"] in payload["recommended_commands"]


def test_adaptive_regression_guard_treats_stable_storage_recovery_floor_as_advisory(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.427,
            "recovery_quality_score": 72.0,
            "backpressure_quality_score": 99.0,
            "backpressure": {
                "total_pending_lines": 3285,
                "core_pending_lines": 1733,
                "deferred_pending_lines": 1552,
                "support_pending_lines": 1,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 102.5,
                "oldest_age_threshold_seconds": 240.0,
            },
            "collector_intake_enforcement_audit": {"status": "enforced", "mismatch_count": 0},
            "backlog_relief_contract": {
                "active": False,
                "p_core_backlog_allocation_contract": {
                    "active": True,
                    "single_writer_only": True,
                    "preprocess_worker_budget": 7,
                    "p_core_burst_intelligence": {"selected_workers": 7},
                    "control_env": {
                        "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
                        "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
                    },
                },
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    floor = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:ingestion_storage_degradation_floor")

    assert payload["overall_status"] == "ready"
    assert floor["state"] == "ready"
    assert floor["adaptive_severity"] == "info"
    assert floor["metrics"]["blockers"] == []
    assert "storage_recovery_quality_below_floor_advisory" in floor["metrics"]["warnings"]
    assert floor["metrics"]["storage_quality_advisory_only"] is True
    assert "backlog_relief_not_active_during_hard_pressure" not in floor["metrics"]["blockers"]


def test_adaptive_regression_guard_treats_clear_storage_backpressure_quality_floor_as_advisory(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.291,
            "recovery_quality_score": 86.0,
            "backpressure_quality_score": 69.26,
            "backpressure": {
                "total_pending_lines": 6943,
                "core_pending_lines": 4372,
                "deferred_pending_lines": 2569,
                "support_pending_lines": 2,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 17.076,
                "oldest_age_threshold_seconds": 240.0,
                "overlay_adjusted": True,
                "overlay_pressure_clear": True,
                "raw_live": {"artifact_stale_for_overlay_reconciliation": False},
            },
            "collector_intake_enforcement_audit": {"status": "partial", "mismatch_count": 1},
            "backlog_relief_contract": {
                "active": True,
                "p_core_backlog_allocation_contract": {
                    "active": True,
                    "single_writer_only": True,
                    "preprocess_worker_budget": 7,
                    "p_core_burst_intelligence": {"selected_workers": 7},
                    "control_env": {
                        "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
                        "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
                    },
                },
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    floor = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:ingestion_storage_degradation_floor")

    assert payload["overall_status"] == "ready"
    assert floor["state"] == "ready"
    assert floor["adaptive_severity"] == "info"
    assert floor["metrics"]["blockers"] == []
    assert floor["metrics"]["storage_operationally_clear"] is True
    assert floor["metrics"]["storage_quality_advisory_only"] is True
    assert "storage_backpressure_quality_below_floor_advisory" in floor["metrics"]["warnings"]
    assert "collector_intake_enforcement_partial" in floor["metrics"]["warnings"]
    assert "collector_intake_not_enforced_during_pressure" not in floor["metrics"]["blockers"]


def test_adaptive_regression_guard_keeps_partial_collector_intake_ready_when_storage_is_clear(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.007,
            "recovery_quality_score": 100.0,
            "backpressure_quality_score": 100.0,
            "backpressure": {
                "total_pending_lines": 2061,
                "core_pending_lines": 71,
                "deferred_pending_lines": 1990,
                "support_pending_lines": 1,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
            },
            "collector_intake_enforcement_audit": {"status": "partial", "mismatch_count": 2},
            "backlog_relief_contract": {
                "active": False,
                "p_core_backlog_allocation_contract": {
                    "active": True,
                    "single_writer_only": True,
                    "preprocess_worker_budget": 7,
                    "p_core_burst_intelligence": {"selected_workers": 7},
                    "control_env": {
                        "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
                        "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
                    },
                },
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    floor = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:ingestion_storage_degradation_floor")

    assert floor["state"] == "ready"
    assert floor["adaptive_severity"] == "info"
    assert floor["metrics"]["blockers"] == []
    assert floor["metrics"]["warnings"] == ["collector_intake_enforcement_partial"]
    assert floor["metrics"]["collector_intake_advisory_only"] is True


def test_adaptive_regression_guard_does_not_block_optional_intake_during_moderate_pressure(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.704,
            "recovery_quality_score": 86.25,
            "backpressure_quality_score": 99.0,
            "backpressure": {
                "total_pending_lines": 3928,
                "core_pending_lines": 3652,
                "deferred_pending_lines": 275,
                "support_pending_lines": 1,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 168.8,
                "oldest_age_threshold_seconds": 240.0,
            },
            "collector_intake_enforcement_audit": {
                "status": "not_required",
                "required": False,
                "mismatch_count": 0,
            },
            "backlog_relief_contract": {"active": False},
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {
            "overall_status": "ready",
            "blocked_surface_count": 0,
            "degraded_surface_count": 0,
            "surfaces": [],
        },
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    floor = next(
        row for row in payload["surfaces"] if row["surface_id"] == "guard:ingestion_storage_degradation_floor"
    )

    assert floor["state"] == "degraded"
    assert floor["metrics"]["collector_intake_optional_safe"] is True
    assert "collector_intake_not_enforced_during_pressure" not in floor["metrics"]["blockers"]
    assert "storage_pressure_index_elevated" in floor["metrics"]["warnings"]


def test_adaptive_regression_guard_manages_verified_bounded_transient_drain(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.704,
            "recovery_quality_score": 86.25,
            "backpressure_quality_score": 99.0,
            "backpressure": {
                "total_pending_lines": 3928,
                "core_pending_lines": 3652,
                "deferred_pending_lines": 275,
                "support_pending_lines": 1,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 168.8,
                "oldest_age_threshold_seconds": 240.0,
            },
            "collector_intake_enforcement_audit": {
                "status": "not_required",
                "required": False,
                "mismatch_count": 0,
            },
            "storage": {"backlog_drain_status": "drain_active"},
            "bounded_recovery_contract": {
                "route_verified": True,
                "active_drain_progress": True,
                "drain_delta_signal_observed": True,
                "hard_gate_active": False,
                "effective_hard_gate_active": False,
            },
            "external_route_verification": {"verification_state": "active_local_ready"},
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "writer_shedding": {"hard_breaches": [], "elevated_breaches": []},
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {
            "overall_status": "ready",
            "blocked_surface_count": 0,
            "degraded_surface_count": 0,
            "surfaces": [],
        },
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    runtime_storage = next(
        row for row in payload["surfaces"] if row["surface_id"] == "guard:runtime_storage_contract"
    )
    floor = next(
        row for row in payload["surfaces"] if row["surface_id"] == "guard:ingestion_storage_degradation_floor"
    )

    assert payload["overall_status"] == "ready"
    assert runtime_storage["state"] == "ready"
    assert runtime_storage["metrics"]["bounded_transient_drain_managed"] is True
    assert runtime_storage["metrics"]["storage_clear"] is True
    assert "storage_pressure_index_elevated" not in runtime_storage["metrics"]["warnings"]
    assert floor["state"] == "ready"
    assert floor["metrics"]["bounded_transient_drain_managed"] is True
    assert floor["metrics"]["storage_operationally_clear"] is True
    assert set(floor["metrics"]["warnings"]) == {
        "storage_recovery_quality_below_target",
        "collector_intake_enforcement_partial",
    }


def test_adaptive_regression_guard_manages_tiny_deferred_drain_without_intake_throttle(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.694,
            "recovery_quality_score": 96.0,
            "backpressure_quality_score": 99.0,
            "backpressure": {
                "total_pending_lines": 35,
                "core_pending_lines": 3,
                "deferred_pending_lines": 32,
                "support_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 166.44,
                "oldest_age_threshold_seconds": 240.0,
            },
            "collector_intake_enforcement_audit": {
                "status": "not_required",
                "required": False,
                "mismatch_count": 0,
            },
            "storage": {"backlog_drain_status": "drain_active"},
            "external_route_verification": {"verification_state": "ready"},
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "writer_shedding": {"hard_breaches": []},
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {
            "overall_status": "ready",
            "blocked_surface_count": 0,
            "degraded_surface_count": 0,
            "surfaces": [],
        },
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    runtime_storage = next(
        row for row in payload["surfaces"] if row["surface_id"] == "guard:runtime_storage_contract"
    )
    floor = next(
        row for row in payload["surfaces"] if row["surface_id"] == "guard:ingestion_storage_degradation_floor"
    )

    assert payload["overall_status"] == "ready"
    assert runtime_storage["state"] == "ready"
    assert runtime_storage["metrics"]["small_residual_drain_managed"] is True
    assert runtime_storage["metrics"]["storage_clear"] is True
    assert "storage_pressure_index_elevated" not in runtime_storage["metrics"]["warnings"]
    assert floor["state"] == "ready"
    assert floor["metrics"]["small_residual_drain_managed"] is True
    assert floor["metrics"]["storage_operationally_clear"] is True
    assert "collector_intake_not_enforced_during_pressure" not in floor["metrics"]["blockers"]
    assert "storage_pressure_index_elevated" not in floor["metrics"]["warnings"]


def test_adaptive_regression_guard_treats_raw_live_reconciliation_stale_as_advisory_when_storage_clear(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.06,
            "recovery_quality_score": 80.0,
            "backpressure_quality_score": 99.0,
            "backpressure": {
                "total_pending_lines": 2708,
                "core_pending_lines": 893,
                "deferred_pending_lines": 1815,
                "support_pending_lines": 70,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
                "raw_live": {"artifact_stale_for_overlay_reconciliation": True},
            },
            "collector_intake_enforcement_audit": {"status": "partial", "mismatch_count": 1},
            "backlog_relief_contract": {
                "active": False,
                "p_core_backlog_allocation_contract": {
                    "active": True,
                    "single_writer_only": True,
                    "preprocess_worker_budget": 7,
                    "p_core_burst_intelligence": {"selected_workers": 7},
                    "control_env": {
                        "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
                        "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
                    },
                },
            },
            "storage_plane_contract": {"disk_contract": {"emergency_disk_guard": False}},
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=tmp_path / "governance" / "health" / "adaptive_regression_guard_state.json",
    )
    floor = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:ingestion_storage_degradation_floor")

    assert floor["state"] == "ready"
    assert floor["metrics"]["storage_operationally_clear"] is True
    assert floor["metrics"]["storage_quality_advisory_only"] is True
    assert "raw_live_backpressure_stale_without_overlay_reconciliation" in floor["metrics"]["warnings"]


def test_adaptive_regression_guard_manages_due_livefeed_route_refresh_while_mirror_is_fresh(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    health = tmp_path / "governance" / "health"
    guard = json.loads((health / "livefeed_refresh_guard_latest.json").read_text(encoding="utf-8"))
    guard["timestamp_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=35)).isoformat()
    _write_json(health / "livefeed_refresh_guard_latest.json", guard)

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=health / "adaptive_regression_guard_state.json",
    )
    livefeed = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:livefeed_visibility_contract")

    assert livefeed["state"] == "ready"
    assert livefeed["metrics"]["refresh_guard_maintenance_due"] is True
    assert livefeed["metrics"]["refresh_guard_staleness_managed"] is True
    assert "livefeed_refresh_guard_stale" not in livefeed["metrics"]["warnings"]


def test_adaptive_regression_guard_does_not_manage_expired_livefeed_route_evidence(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    health = tmp_path / "governance" / "health"
    guard = json.loads((health / "livefeed_refresh_guard_latest.json").read_text(encoding="utf-8"))
    guard["timestamp_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=61)).isoformat()
    _write_json(health / "livefeed_refresh_guard_latest.json", guard)

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=health / "adaptive_regression_guard_state.json",
    )
    livefeed = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:livefeed_visibility_contract")

    assert livefeed["state"] == "degraded"
    assert livefeed["metrics"]["refresh_guard_staleness_managed"] is False
    assert "livefeed_refresh_guard_stale" in livefeed["metrics"]["warnings"]


def test_adaptive_regression_guard_accepts_external_pressure_when_paper_contract_is_armed(tmp_path: Path) -> None:
    _seed_ready_artifacts(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "advisory",
            "throttle_profile": "sustain",
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "host_saturation_score": 82.0,
            "host_pressure_attribution": {
                "external_pressure_dominant": True,
                "bot_owned_pressure_dominant": False,
            },
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "reason": "paper_ramp_armed_and_clean",
            },
            "soft_cap_advisory_reclassification": {
                "active": True,
                "to_status": "advisory",
                "reason": "research_training_pressure_is_already_niced_and_guarded_advisory",
            },
        },
    )
    _write_json(
        health / "paper_400_ramp_latest.json",
        {
            "timestamp_utc": now,
            "ok": True,
            "stage": "armed",
            "armed": True,
            "blockers": [],
            "gates": {
                "runtime": {
                    "ok": True,
                    "status": "ready",
                    "blockers": [],
                    "runtime_capacity_ready": True,
                    "capacity_limited_armed": True,
                    "paper_execution_clean": True,
                    "live_execution_locked": True,
                }
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        grade_guard_builder=lambda _: {"overall_status": "ready", "blocked_surface_count": 0, "degraded_surface_count": 0, "surfaces": []},
        state_path=health / "adaptive_regression_guard_state.json",
    )
    runtime_storage = next(row for row in payload["surfaces"] if row["surface_id"] == "guard:runtime_storage_contract")

    assert runtime_storage["state"] == "ready"
    assert runtime_storage["metrics"]["external_pressure_advisory"] is True
    assert runtime_storage["metrics"]["external_paper_soak_advisory"] is True
    assert runtime_storage["metrics"]["paper_execution_open"] is True
    assert "host_saturation_elevated" not in runtime_storage["metrics"]["warnings"]
