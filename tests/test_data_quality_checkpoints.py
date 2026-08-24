from core.data_quality_checkpoints import run_checkpoint

EXPECTATIONS = {
    "required_columns": ["event_id", "event_time_utc", "price"],
    "types": {"event_id": "string", "event_time_utc": "timestamp", "price": "number"},
    "ranges": {"price": {"min": 0.01}},
    "unique_columns": ["event_id"],
    "monotonic_columns": ["event_time_utc"],
    "freshness": {"column": "event_time_utc", "max_age_seconds": 120},
    "failure_action": "block",
}


def test_declarative_checkpoint_passes_fresh_valid_rows() -> None:
    report = run_checkpoint(
        checkpoint_id="market-events",
        rows=[
            {
                "event_id": "1",
                "event_time_utc": "2026-08-21T11:59:00+00:00",
                "price": 100.0,
            },
            {
                "event_id": "2",
                "event_time_utc": "2026-08-21T12:00:00+00:00",
                "price": 101.0,
            },
        ],
        expectations=EXPECTATIONS,
        observed_at_utc="2026-08-21T12:00:30+00:00",
    )

    assert report["ok"] is True
    assert report["status"] == "passed"
    assert report["great_expectations_runtime_installed"] is False


def test_declarative_checkpoint_blocks_and_quarantines_bad_rows() -> None:
    report = run_checkpoint(
        checkpoint_id="market-events",
        rows=[
            {
                "event_id": "1",
                "event_time_utc": "2026-08-21T11:00:00+00:00",
                "price": -1.0,
            },
            {
                "event_id": "1",
                "event_time_utc": "2026-08-21T10:00:00+00:00",
                "price": "bad",
            },
        ],
        expectations=EXPECTATIONS,
        observed_at_utc="2026-08-21T12:00:30+00:00",
    )

    assert report["ok"] is False
    assert report["status"] == "blocked"
    assert report["quarantine_row_indices"] == [0, 1]
    assert report["execution_authority"] is False
