import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import writer_process_intelligence as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_writer_process_intelligence_scores_stale_writer_and_expanded_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "writer_state_before": {
                "active": True,
                "running": True,
                "current_step": "merge_primary",
                "progress_age_minutes": 61.0,
                "cycle_age_minutes": 88.0,
                "merged_rows_this_cycle": 3476,
                "completed_merge_count": 18,
                "writer_lock_owner": "pid=123 cmd=sql_link_shard_manager",
            },
            "summary": {
                "writer_progress_observed": False,
                "stale_writer_detected": True,
            },
        },
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {"overall_status": "ready", "status": [{"name": "sql_link_writer", "running": 1, "raw_running": 1}]},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"overall_status": "blocked", "severity": "critical", "total_pending_lines": 17267},
    )
    _write_json(
        health / "drainer_intelligence_layer_latest.json",
        {"overall_status": "ready", "decision_packet": {"action": "verify_writer_progress_then_re_score"}},
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "advisory"
    assert payload["decision_packet"]["action"] == "verify_writer_progress_then_re_score"
    assert payload["writer_health"]["state"] == "stale_progress"
    assert payload["decision_packet"]["expanded_writer_lane_count"] >= 25
    assert payload["safety_envelope"]["single_writer_only"] is True
    assert payload["safety_envelope"]["starts_parallel_sql_writers"] is False
    families = {row["family"] for row in payload["lane_family_summary"]}
    assert "admission_evidence" in families
    assert "writer_health" in families


def test_writer_process_intelligence_blocks_duplicate_sql_writers(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "ready",
            "writer_state_after_wait": {"active": False, "current_step": "complete"},
        },
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {"overall_status": "degraded", "status": [{"name": "sql_link_writer", "running": 2, "raw_running": 2}]},
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["decision_packet"]["action"] == "enforce_single_writer_process_then_re_score"
    assert "duplicate_sql_writer_processes" in payload["decision_packet"]["risk_flags"]


def test_writer_process_intelligence_does_not_wait_on_orphaned_progress(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    locks = project_root / "governance" / "locks"
    health.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "ready",
            "writer_state_before": {
                "active": False,
                "active_source": "orphaned_progress",
                "progress_orphaned": True,
                "running": True,
                "current_step": "shard_linking",
                "progress_age_minutes": 5.0,
                "writer_lock_owner": "",
                "writer_lock_held": False,
            },
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": "2026-05-02T11:55:00+00:00",
            "status": "running",
            "running": True,
            "current_step": "shard_linking",
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"overall_status": "blocked", "severity": "critical", "total_pending_lines": 25000},
    )
    (locks / "jsonl_sql_writer.lock").write_text("", encoding="utf-8")

    payload = src.build_payload(project_root)

    assert payload["writer_health"]["active"] is False
    assert payload["writer_health"]["state"] == "orphaned_progress"
    assert payload["decision_packet"]["action"] == "run_focused_writer_cycle"
    assert "writer_progress_orphaned" in payload["decision_packet"]["risk_flags"]
    assert "writer_active" not in payload["decision_packet"]["risk_flags"]


def test_writer_process_intelligence_does_not_trust_dead_writer_owner_pid(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "apply_failed",
            "writer_state_after_wait": {
                "active": True,
                "active_source": "recent_progress",
                "progress_orphaned": False,
                "writer_owner_pid_live": False,
                "running": True,
                "current_step": "shard_linking",
                "progress_age_minutes": 6.0,
                "writer_lock_owner": "pid=999999 started=old cmd=sql_link_shard_manager",
                "writer_lock_held": False,
            },
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": "2026-05-02T11:54:00+00:00",
            "status": "running",
            "running": True,
            "current_step": "shard_linking",
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"overall_status": "blocked", "severity": "critical", "total_pending_lines": 25000},
    )

    payload = src.build_payload(project_root)

    assert payload["writer_health"]["active"] is False
    assert payload["writer_health"]["state"] == "orphaned_progress"
    assert payload["decision_packet"]["action"] == "run_focused_writer_cycle"


def test_writer_process_intelligence_reports_child_writer_work_after_complete_step(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "writer_state_before": {
                "active": True,
                "running": True,
                "current_step": "complete",
                "effective_current_step": "shard_worker_active_after_reported_complete",
                "child_writer_active": True,
                "active_child_writer_count": 1,
                "active_child_writer_pids": [456],
                "progress_age_minutes": 1.0,
                "cycle_age_minutes": 5.0,
                "merged_rows_this_cycle": 1200,
                "completed_merge_count": 25,
                "writer_lock_owner": "pid=123 cmd=sql_link_shard_manager",
                "writer_lock_held": True,
            },
        },
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {"overall_status": "ready", "status": [{"name": "sql_link_writer", "running": 1, "raw_running": 1}]},
    )

    payload = src.build_payload(project_root)

    assert payload["writer_health"]["state"] == "active_progressing"
    assert payload["writer_health"]["current_step"] == "shard_worker_active_after_reported_complete"
    assert payload["writer_health"]["child_writer_active"] is True
    assert payload["writer_health"]["active_child_writer_pids"] == [456]


def test_writer_process_intelligence_surfaces_shard_link_plan_and_timeouts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "writer_state_before": {
                "active": True,
                "running": True,
                "current_step": "shard_linking",
                "progress_age_minutes": 1.0,
                "cycle_age_minutes": 3.0,
                "completed_shard_count": 2,
                "planned_shard_count": 4,
                "pending_shard_count": 2,
                "timed_out_shard_count": 1,
                "writer_lock_owner": "pid=123 cmd=sql_link_shard_manager",
                "writer_lock_held": True,
            },
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": "2026-05-26T13:00:00+00:00",
            "status": "running",
            "running": True,
            "current_step": "shard_linking",
            "planned_shard_count": 4,
            "completed_shard_count": 2,
            "pending_shard_count": 2,
            "timed_out_shard_count": 1,
            "pending_shards": ["data", "explanations"],
            "timed_out_shards": ["trading"],
            "shard_link_plan": {
                "policy": "adaptive_hot_pending_sentinel_first",
                "planned_order": ["health_fast", "trading", "data", "explanations"],
            },
        },
    )

    payload = src.build_payload(project_root)

    assert payload["writer_health"]["planned_shard_count"] == 4
    assert payload["writer_health"]["pending_shards"] == ["data", "explanations"]
    assert payload["writer_health"]["timed_out_shards"] == ["trading"]
    assert payload["writer_health"]["shard_link_plan_policy"] == "adaptive_hot_pending_sentinel_first"
    assert "shard_link_timeouts" in payload["decision_packet"]["risk_flags"]


def test_writer_process_intelligence_distinguishes_idle_service_lock_from_active_progress(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "idle",
            "writer_state_before": {
                "active": True,
                "active_source": "writer_lock",
                "running": False,
                "current_step": "complete",
                "progress_age_minutes": 1.0,
                "cycle_age_minutes": 5.0,
                "merged_rows_this_cycle": 1200,
                "completed_merge_count": 25,
                "writer_lock_owner": "pid=123 cmd=sql_link_shard_manager",
                "writer_lock_held": True,
            },
        },
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {"overall_status": "ready", "status": [{"name": "sql_link_writer", "running": 1, "raw_running": 1}]},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"overall_status": "ready", "severity": "stable", "backpressure": {"total_pending_lines": 25000}},
    )

    payload = src.build_payload(project_root)

    assert payload["writer_health"]["state"] == "service_idle_holding_lock"
    assert payload["decision_packet"]["action"] == "request_writer_service_handoff_then_re_score"
    assert "writer_service_idle_lock" in payload["decision_packet"]["risk_flags"]
    assert "writer_active" not in payload["decision_packet"]["risk_flags"]
