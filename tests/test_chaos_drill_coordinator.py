from pathlib import Path

from scripts.ops import chaos_drill_coordinator as src


def test_isolated_harness_records_all_weekly_recovery_drills(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "chaos_state.json"
    harness_path = tmp_path / "health" / "production_recovery_drill_harness_latest.json"

    harness = src._record_isolated_harness(
        tmp_path,
        state_path=state_path,
        harness_path=harness_path,
    )
    payload = src.build_payload(tmp_path, state_path=state_path, overdue_days=7.0)

    assert harness["ok"] is True
    assert harness_path.exists()
    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["verified_drill_count"] == payload["required_drill_count"] == 10
    assert payload["overdue_drills"] == []
    assert all(row["evidence_class"] == "deterministic_isolated_recovery_drill" for row in payload["drills"])
    assert all(row["real_outage_evidence"] is False for row in payload["drills"])
