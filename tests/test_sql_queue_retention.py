from scripts import sql_queue_retention as src


def test_queue_retention_uses_shorter_window_under_capacity_pressure() -> None:
    hours, mode = src._effective_acked_retention_hours(
        db_size_gb=12.0,
        acked_days=7,
        acked_hours=-1.0,
        pressure_db_gb=8.0,
        pressure_acked_hours=6.0,
    )

    assert hours == 6.0
    assert mode == "capacity_pressure"


def test_queue_retention_explicit_hours_take_precedence() -> None:
    hours, mode = src._effective_acked_retention_hours(
        db_size_gb=12.0,
        acked_days=7,
        acked_hours=0.25,
        pressure_db_gb=8.0,
        pressure_acked_hours=6.0,
    )

    assert hours == 0.25
    assert mode == "explicit_hours"
