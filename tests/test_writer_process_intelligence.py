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
