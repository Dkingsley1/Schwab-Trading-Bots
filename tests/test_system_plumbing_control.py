import json
from pathlib import Path

from scripts.ops import system_plumbing_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_plumbing(
    project_root: Path,
    *,
    execution_expected: bool = False,
    raw_total: int = 2172,
    pressure_index: float = 0.137,
    overlay_total: int | None = None,
    storage_status: str = "ready",
    storage_severity: str = "stable",
) -> None:
    health = project_root / "governance" / "health"
    _write_json(health / "process_watchdog_latest.json", {"overall_status": "ready", "alerts": [], "safety_pause": {"active": False}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": storage_status,
            "severity": storage_severity,
            "pressure_index": pressure_index,
            "backpressure_quality_score": 100,
            "backpressure": {
                "core_pending_lines": overlay_total if overlay_total is not None else raw_total,
                "total_pending_lines": overlay_total if overlay_total is not None else raw_total,
                "overlay_adjusted": overlay_total is not None,
                "overlay_pressure_clear": overlay_total is not None,
                "raw_live": {
                    "core_pending_lines": min(raw_total, 6000),
                    "total_pending_lines": raw_total,
                    "oldest_pending_age_seconds": 23.5,
                },
            },
            "external_route_verification": {
                "verification_state": "active_local_ready",
                "ready_count": 3,
                "tracked_count": 3,
                "coverage_ratio": 1.0,
                "mismatches": [],
            },
            "storage_resilience": {"unresolved_split_brain_conflicts": 0},
            "storage_plane_contract": {"disk_contract": {"external_disk": {"exists": True, "available_gb": 62.0}}},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {"overall_status": "blocked", "blocked_reasons": ["external_storage_unavailable"]},
    )
    _write_json(health / "storage_failback_sync_latest.json", {"split_brain_conflicts": 0})
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "writer_health": {
                "state": "active_progressing",
                "active": True,
                "current_step": "shard_linking",
                "completed_shard_count": 13,
                "planned_shard_count": 26,
                "timed_out_shard_count": 0,
                "active_child_writer_count": 1,
            },
            "process_topology": {"duplicate_sql_writer_processes": False},
            "safety_envelope": {"single_writer_only": True},
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "waiting_for_writer"})
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 6,
            "raw_write_failure_count": 6,
            "account_snapshot_failure_count": 0,
            "queue_depth": raw_total,
            "current_storage_write_ready": True,
            "writer_handoff_contract": {"writer_service_active": True},
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {"core": {"pending_lines": raw_total}}})
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "memory_pressure_level": "normal"})
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "global_halt_auto_clear_latest.json",
        {
            "halt": False,
            "halt_state": "clear_blocked",
            "clear_blockers": ["write_path_recovery_pending"],
            "metrics": {"execution_expected": execution_expected},
        },
    )


def test_system_plumbing_relieves_bounded_write_path_recovery_for_paper(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_plumbing(project_root)

    payload = src.build_payload(project_root)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["sections"]["data_plane_recovery"]["bounded_write_recovery"] is True
    assert payload["global_clear_relief"]["active"] is True
    assert payload["global_clear_relief"]["bounded_write_recovery"] is True
    assert payload["paper_ramp_relief_contract"]["bounded_write_recovery"] is True
    assert "external_backlog_drain_advisory" in payload["warnings"]
    assert payload["plumbing_score"] == 100
    assert set(payload["managed_advisories"]["managed"]) == {
        "external_backlog_drain_advisory",
        "external_storage_reserve_advisory",
        "write_path_recovery_advisory",
    }
    assert payload["managed_advisories"]["unmanaged"] == []


def test_system_plumbing_blocks_write_relief_when_live_execution_is_expected(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_plumbing(project_root, execution_expected=True)

    payload = src.build_payload(project_root)

    assert payload["ok"] is False
    assert "execution_boundary_blocked" in payload["blockers"]
    assert "global_clear_blockers_unbounded" in payload["blockers"]
    assert payload["global_clear_relief"]["active"] is False


def test_system_plumbing_blocks_when_raw_live_queue_is_not_clear(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_plumbing(project_root, raw_total=22000)

    payload = src.build_payload(project_root)

    assert payload["ok"] is False
    assert "queue_backpressure_blocked" in payload["blockers"]
    assert payload["sections"]["queue_backpressure"]["raw_live"]["ok"] is False
    assert payload["root_cause"]["primary"] == "queue_backpressure_blocked"


def test_system_plumbing_treats_overlay_pressure_band_as_paper_advisory(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_plumbing(project_root, pressure_index=0.334, raw_total=2172, overlay_total=5156)

    payload = src.build_payload(project_root)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["sections"]["queue_backpressure"]["status"] == "storage_pressure_advisory"
    assert payload["sections"]["queue_backpressure"]["pressure_advisory"] is True
    assert payload["sections"]["queue_backpressure"]["overlay_relief"]["active"] is True
    assert "storage_pressure_hysteresis_advisory" in payload["warnings"]
    assert "sql_overlay_cleanup_advisory" in payload["warnings"]
    assert payload["plumbing_score"] == 100
    assert payload["managed_advisories"]["all_managed"] is True
    assert payload["root_cause"]["status"] == "advisory"
    assert payload["paper_ramp_relief_contract"]["bounded_write_recovery"] is True


def test_system_plumbing_treats_sql_overlay_only_critical_storage_as_advisory(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_plumbing(
        project_root,
        pressure_index=22.88,
        raw_total=2172,
        overlay_total=2717,
        storage_status="blocked",
        storage_severity="critical",
    )

    payload = src.build_payload(project_root)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["sections"]["queue_backpressure"]["pressure_hard"] is False
    assert payload["sections"]["queue_backpressure"]["overlay_relief"]["active"] is True
    assert "queue_backpressure_blocked" not in payload["blockers"]
    assert "sql_overlay_cleanup_advisory" in payload["warnings"]
    assert payload["plumbing_score"] == 100
    assert payload["managed_advisories"]["all_managed"] is True
    assert payload["root_cause"]["status"] == "advisory"


def test_system_plumbing_manages_deferred_off_hours_backlog_for_paper(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_plumbing(project_root, raw_total=38667)
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 53.028,
            "backpressure_quality_score": 40,
            "backpressure": {
                "core_pending_lines": 809,
                "support_pending_lines": 4246,
                "deferred_pending_lines": 15899259,
                "total_pending_lines": 15904314,
                "pending_lines_threshold": 15000,
            },
            "storage": {"backlog_drain_status": "waiting_for_off_hours"},
            "external_route_verification": {
                "verification_state": "ready",
                "ready_count": 3,
                "tracked_count": 3,
                "coverage_ratio": 1.0,
                "mismatches": [],
            },
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "writer_shedding": {"hard_breaches": ["deferred"], "elevated_breaches": ["core", "deferred"]},
            "storage_resilience": {"unresolved_split_brain_conflicts": 0},
            "storage_plane_contract": {"disk_contract": {"external_disk": {"exists": True, "available_gb": 560.0}}},
        },
    )
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 2,
            "raw_write_failure_count": 2,
            "account_snapshot_failure_count": 0,
            "queue_depth": 38667,
            "writer_handoff_contract": {"writer_service_active": True},
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {"core": {"pending_lines": 38667}}})
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "blocked",
            "compute_pressure_level": "normal",
            "memory_pressure_level": "normal",
            "host_saturation_score": 27.0,
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "blocked",
            "memory_snapshot": {
                "memory_free_pct": 90.0,
                "swap_used_gb": 2.4,
                "compressed_store_gb": 6.2,
                "compressor_gb": 0.3,
            },
        },
    )
    _write_json(
        health / "global_halt_auto_clear_latest.json",
        {
            "halt": False,
            "halt_required": True,
            "would_rehalt": True,
            "halt_posture": "unlatched_halt_required",
            "clear_blockers": ["write_path_recovery_pending", "queue_backpressure_active"],
            "metrics": {"execution_expected": False},
        },
    )

    payload = src.build_payload(project_root)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["sections"]["queue_backpressure"]["status"] == "managed_deferred_backlog_advisory"
    assert payload["sections"]["runtime_memory"]["status"] == "managed_deferred_backlog_advisory"
    assert payload["sections"]["data_plane_recovery"]["bounded_write_recovery"] is True
    assert payload["global_clear_relief"]["status"] == "managed_deferred_backpressure_advisory"
    assert payload["paper_ramp_relief_contract"]["managed_deferred_backlog"] is True
    assert "managed_deferred_backlog_advisory" in payload["warnings"]
    assert payload["managed_advisories"]["all_managed"] is True


def test_system_plumbing_consumes_runtime_external_high_compute_relief_for_paper(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_plumbing(project_root)
    health = project_root / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "advisory",
            "host_saturation_score": 50.0,
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "soft_cap_advisory_reclassification": {
                "active": True,
                "to_status": "advisory",
                "reason": "external_high_compute_with_bounded_storage_overlay_is_capacity_limited_advisory",
                "thresholds": {"max_guarded_external_high_compute_host_saturation_score": 75.0},
                "measurements": {
                    "external_high_compute_guarded": True,
                    "bounded_storage_overlay_guarded": True,
                    "paper_ramp_memory_guarded": True,
                    "paper_execution_hot": False,
                    "bot_owned_pressure_dominant": False,
                    "host_saturation_score": 50.0,
                },
            },
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "armed": True,
                "ok": True,
            },
        },
    )
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "needs_work"})

    payload = src.build_payload(project_root)
    runtime = payload["sections"]["runtime_memory"]

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert "runtime_memory_blocked" not in payload["blockers"]
    assert runtime["status"] == "advisory"
    assert runtime["paper_only_runtime_memory_relief"] is True
    assert runtime["runtime_soft_cap_paper_relief"]["ok"] is True
    assert "compute_pressure_advisory" in payload["warnings"]
    assert "runtime_memory_paper_advisory" in payload["warnings"]
    assert payload["managed_advisories"]["all_managed"] is True


def test_system_plumbing_score_still_penalizes_hard_blockers(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_plumbing(project_root, raw_total=22000)

    payload = src.build_payload(project_root)

    assert payload["ok"] is False
    assert "queue_backpressure_blocked" in payload["blockers"]
    assert payload["plumbing_score"] < 100
