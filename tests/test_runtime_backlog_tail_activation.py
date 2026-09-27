import pytest

from scripts.ops import runtime_throttle_control as governor


@pytest.mark.parametrize(
    "core,total,age",
    [
        (1001, 1500, 10),
        (10, 2501, 10),
        (20, 50, 61),
    ],
)
@pytest.mark.parametrize("profile", ["observe", "soft_cap", "sustain"])
def test_small_debt_uses_admitted_accelerator_budget(profile, core, total, age):
    result = governor._sql_overrides_for_runtime_pressure(
        profile,
        storage_drain_active=True,
        storage_pressure=dict(
            core_pending_lines=core,
            total_pending_lines=total,
            oldest_pending_age_seconds=age,
            pressure_index=0,
        ),
        sql_writer_coordination={},
        writer_worker_budget=3,
        max_writer_lanes=3,
    )
    assert result["SQL_LINK_SERVICE_SHARD_WRITER_LANES"] == "3"
    assert result["SQL_LINK_SERVICE_INTERVAL_SECONDS"] == "12"


def test_near_empty_queue_and_hard_pressure_keep_single_lane():
    for profile, core, total, age in [
        ("sustain", 1000, 2500, 60),
        ("protect_live", 1001, 2501, 61),
    ]:
        result = governor._sql_overrides_for_runtime_pressure(
            profile,
            storage_drain_active=True,
            storage_pressure=dict(
                core_pending_lines=core,
                total_pending_lines=total,
                oldest_pending_age_seconds=age,
                pressure_index=0,
            ),
            sql_writer_coordination={},
            writer_worker_budget=3,
            max_writer_lanes=3,
        )
        assert result["SQL_LINK_SERVICE_SHARD_WRITER_LANES"] == "1"


def test_aged_empty_queue_does_not_trigger_new_tail_acceleration():
    assert not governor._storage_drain_requires_acceleration(
        dict(
            core_pending_lines=0,
            total_pending_lines=0,
            oldest_pending_age_seconds=90,
            pressure_index=0,
        ),
        {},
        responsive_tail=True,
    )


def test_tail_acceleration_preserves_single_worker_admission():
    result = governor._sql_overrides_for_runtime_pressure(
        "sustain",
        storage_drain_active=True,
        storage_pressure=dict(
            core_pending_lines=1500,
            total_pending_lines=3000,
            oldest_pending_age_seconds=90,
            pressure_index=0,
        ),
        sql_writer_coordination={},
        writer_worker_budget=1,
        max_writer_lanes=1,
    )
    assert result["SQL_LINK_SERVICE_SHARD_WRITER_LANES"] == "1"
