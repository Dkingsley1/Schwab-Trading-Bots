import json
from datetime import datetime, timezone

from scripts.ops import self_healing_gap_audit as audit

NOW = datetime(2026, 9, 23, 12, tzinfo=timezone.utc)


def test_census_preserves_stale_truth_and_never_runs_commands(tmp_path, monkeypatch):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    stale = {
        "timestamp_utc": "2026-09-20T12:00:00+00:00",
        "next_best_command": "do-not-run",
    }
    path = health / "jsonl_sql_ingestion_health_test_latest.json"
    path.write_text(json.dumps(stale))
    (health / "future_latest.json").write_text(
        '{"timestamp_utc":"2026-09-24T00:00:00Z"}'
    )
    (health / "bad_latest.json").write_text("not json")
    (health / "link_latest.json").symlink_to(path)
    monkeypatch.setattr(
        audit,
        "declared_intake_catalog",
        lambda root: {
            "collectors": [
                {"producer_id": "optional", "collector_contract": {"required": False}},
                {"producer_id": "required", "collector_contract": {"required": True}},
            ],
            "artifact_producers": [
                {
                    "producer_id": "test",
                    "payload": {"relative_path": str(path.relative_to(tmp_path))},
                    "owner_command": None,
                    "max_age_minutes": 60,
                }
            ],
        },
    )
    result = audit.build(tmp_path, now=NOW)
    assert result["scan_complete"] and result["stale_or_unverified_count"] == 4
    assert result["sql_overlay"][0]["age_hours"] == 72
    assert result["repair_command_references"][0]["command"] == "do-not-run"
    assert result["artifact_producers"][0]["missing_owner_alert"]
    assert result["artifact_producers"][0]["owner_refresh_overdue"]
    assert (
        result["collectors"][0]["proof_requirement"]
        == "advisory_only_not_execution_evidence"
    )
    assert result["collectors"][1]["proof_requirement"] == "runtime_evidence_required"
    assert not result["fresh_timestamp_is_ingestion_proof"]
    assert json.loads(path.read_text()) == stale


def test_bounded_census_marks_incomplete(tmp_path, monkeypatch):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    (health / "a_latest.json").write_text("{}")
    monkeypatch.setattr(
        audit,
        "declared_intake_catalog",
        lambda root: {"collectors": [], "artifact_producers": []},
    )
    result = audit.build(tmp_path, now=NOW, seconds=0)
    assert not result["scan_complete"] and not result["ok"]
    assert result["incomplete_reasons"] == ["census_deadline"]


def test_payload_read_is_size_bounded(tmp_path):
    (tmp_path / "large.json").write_text("x" * 20)
    assert audit.read(tmp_path, "large.json", max_bytes=10) == ({}, "size_budget")


def test_source_age_wins_over_new_report_timestamp():
    payload = {"timestamp_utc": NOW.isoformat(), "source_timestamp_utc": "2026-09-20T12:00:00Z"}
    assert audit.age(payload, NOW) == (72, "stale_over_24h")
    payload["source_timestamp_utc"] = "invalid"
    assert audit.age(payload, NOW)[1] == "timestamp_invalid"


def test_producer_budget_uses_exact_age_not_generic_day():
    payload = {"timestamp_utc": "2026-09-23T11:44:59Z"}
    assert audit.age(payload, NOW, 900)[1] == "stale"


def test_fifo_cannot_block_census(tmp_path):
    import os
    os.mkfifo(tmp_path / "pipe.json")
    assert audit.read(tmp_path, "pipe.json") == ({}, "not_regular")
