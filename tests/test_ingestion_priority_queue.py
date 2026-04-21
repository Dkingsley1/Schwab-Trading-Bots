import json
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import ingestion_priority_queue as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_ingestion_priority_queue_persists_lane_quotas_and_retry_state(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "top_pending_files": [{"source_rel": "decisions/a.jsonl", "pending_lines": 1200, "total_lines": 5000, "oldest_pending_age_seconds": 90, "last_line": 3800}],
            "top_deferred_pending_files": [{"source_rel": "governance/events/b.jsonl", "pending_lines": 800, "total_lines": 800, "oldest_pending_age_seconds": 30, "last_line": 0}],
            "top_cold_pending_files": [{"source_rel": "governance/shadow/c.jsonl", "pending_lines": 400, "total_lines": 1000, "oldest_pending_age_seconds": 15, "last_line": 600}],
        },
    )
    db_path = tmp_path / "queue.sqlite3"
    payload = src.build_payload(tmp_path, db_path=db_path, top_n=5)
    assert payload["queue_depth"] == 3
    assert payload["lane_counts"]["core"]["quota_share"] == 0.6
    assert payload["lane_counts"]["core"]["adaptive_quota_share"] > 0.0
    dispatch = {row["source_rel"]: row for row in payload["dispatch_plan"]}
    assert dispatch["decisions/a.jsonl"]["replay_from_line"] == 3800
    assert dispatch["decisions/a.jsonl"]["replay_to_line"] == 5000
    assert dispatch["governance/events/b.jsonl"]["replay_from_line"] == 0
    assert dispatch["governance/events/b.jsonl"]["replay_to_line"] == 800

    with sqlite3.connect(str(db_path)) as conn:
        assert src._mark_retry(conn, "decisions/a.jsonl") is True

    payload = src.build_payload(tmp_path, db_path=db_path, top_n=5)
    retry_candidates = payload["retry_candidates"]
    assert retry_candidates
    assert retry_candidates[0]["retry_count"] == 1
    assert retry_candidates[0]["fairness_bonus"] > 0.0
    assert retry_candidates[0]["effective_priority_score"] > retry_candidates[0]["priority_score"]


def test_ingestion_priority_queue_ack_sets_replay_cursor_complete(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "top_pending_files": [{"source_rel": "decisions/a.jsonl", "pending_lines": 100, "total_lines": 250, "oldest_pending_age_seconds": 20, "last_line": 150}],
        },
    )
    db_path = tmp_path / "queue.sqlite3"
    src.build_payload(tmp_path, db_path=db_path, top_n=5)
    with sqlite3.connect(str(db_path)) as conn:
        assert src._ack(conn, "decisions/a.jsonl") is True
    payload = src.build_payload(tmp_path, db_path=db_path, top_n=5)
    acked = payload["acked_items"]
    assert acked == []


def test_ingestion_priority_queue_replay_window_targets_pending_tail(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "top_pending_files": [{"source_rel": "decisions/a.jsonl", "pending_lines": 125, "total_lines": 400, "oldest_pending_age_seconds": 20, "last_line": 275}],
        },
    )

    payload = src.build_payload(tmp_path, db_path=tmp_path / "queue.sqlite3", top_n=5)
    row = payload["dispatch_plan"][0]
    assert row["replay_from_line"] == 275
    assert row["replay_to_line"] == 400
