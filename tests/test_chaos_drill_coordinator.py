from pathlib import Path
import json

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


def test_isolated_harness_cadence_prevents_repeated_heavy_runs(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "chaos_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"drills": {}, "last_isolated_harness": {"timestamp_utc": src.iso_now()}}),
        encoding="utf-8",
    )

    cadence = src._isolated_run_due(state_path, min_interval_hours=24.0)
    forced = src._isolated_run_due(state_path, min_interval_hours=24.0, force=True)

    assert cadence["due"] is False
    assert cadence["reason"] == "cadence_guard_active"
    assert forced["due"] is True


def test_recorded_recovery_time_above_slo_fails_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "chaos_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "drills": {
                    "auth_expiry": {
                        "completed_at_utc": src.iso_now(),
                        "result": "pass",
                        "containment_verified": True,
                        "no_duplicate_orders": True,
                        "recovery_seconds": 31.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path, state_path=state_path, max_recovery_seconds=30.0)

    assert payload["overall_status"] == "blocked"
    assert [row["drill"] for row in payload["recovery_slo_breaches"]] == ["auth_expiry"]
