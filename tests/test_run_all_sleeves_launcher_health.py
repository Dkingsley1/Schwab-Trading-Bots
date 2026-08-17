from __future__ import annotations

from datetime import datetime, timezone

from scripts import run_all_sleeves as src


class DummyProc:
    def __init__(self, pid: int, exit_code: int | None) -> None:
        self.pid = pid
        self._exit_code = exit_code

    def poll(self) -> int | None:
        return self._exit_code


def _spec(name: str) -> src.JobSpec:
    return src.JobSpec(name=name, cmd=["python", "-c", "pass"], env={}, breaker_group="test")


def test_launcher_health_ready_when_non_running_jobs_cleanly_exited() -> None:
    specs = {name: _spec(name) for name in ("core", "specialized", "dividend")}
    procs = {
        "core": DummyProc(101, None),
        "specialized": DummyProc(102, 0),
        "dividend": DummyProc(103, 0),
    }

    payload = src._launcher_health_payload(
        specs=specs,
        procs=procs,  # type: ignore[arg-type]
        proc_started_at={"core": 1.0, "specialized": 1.0, "dividend": 1.0},
        restart_history={},
        quarantined_jobs={},
        launcher_started_at=1.0,
        phase="running",
        clean_exited_jobs={"specialized", "dividend"},
    )

    assert payload["overall_status"] == "ready"
    assert payload["repair_packet"]["status"] == "clear"
    assert payload["clean_exited_job_count"] == 2
    contract = payload["launcher_readiness_contract"]
    assert contract["mode"] == "sleeve_launcher_readiness_expansion_v2"
    assert contract["paper_only"] is True
    assert contract["live_execution_allowed"] is False
    assert contract["readiness_status"] == "stable_with_parked_lanes"
    assert contract["can_expand_collection_sleeves"] is True
    assert contract["max_new_collect_only_sleeves"] == 3
    assert "sleeve_expansion_admission_infrabot" in [row["name"] for row in payload["repair_infrabots"]]


def test_launcher_health_still_degraded_for_unexpected_exit() -> None:
    specs = {name: _spec(name) for name in ("core", "specialized")}
    procs = {
        "core": DummyProc(101, None),
        "specialized": DummyProc(102, 1),
    }

    payload = src._launcher_health_payload(
        specs=specs,
        procs=procs,  # type: ignore[arg-type]
        proc_started_at={"core": 1.0, "specialized": 1.0},
        restart_history={},
        quarantined_jobs={},
        launcher_started_at=1.0,
        phase="running",
        clean_exited_jobs=set(),
    )

    assert payload["overall_status"] == "degraded"
    assert payload["repair_packet"]["status"] == "needs_repair"
    assert payload["repair_packet"]["problem_job_count"] == 1
    contract = payload["launcher_readiness_contract"]
    assert contract["readiness_status"] == "repair_optional_first"
    assert contract["problem_job_count"] == 1
    assert contract["exact_needs"][0]["target"] == "specialized"
    assert contract["max_new_collect_only_sleeves"] == 0


def test_launcher_health_counts_unspawned_policy_parked_executor_as_stable() -> None:
    specs = {name: _spec(name) for name in ("baseline_parallel", "paper_executor")}
    procs = {"baseline_parallel": DummyProc(101, None)}

    payload = src._launcher_health_payload(
        specs=specs,
        procs=procs,  # type: ignore[arg-type]
        proc_started_at={"baseline_parallel": 1.0},
        restart_history={},
        quarantined_jobs={},
        launcher_started_at=1.0,
        phase="running",
        policy_parked_jobs={"paper_executor"},
    )

    paper_job = next(row for row in payload["jobs"] if row["name"] == "paper_executor")
    contract = payload["launcher_readiness_contract"]

    assert payload["overall_status"] == "ready"
    assert payload["expected_job_count"] == 2
    assert payload["running_job_count"] == 1
    assert payload["missing_job_count"] == 0
    assert payload["policy_parked_job_count"] == 1
    assert paper_job["state"] == "policy_parked"
    assert paper_job["policy_parked"] is True
    assert contract["readiness_status"] == "stable_with_parked_lanes"
    assert contract["class_counts"]["execution_lane"]["stable_non_running"] == 1


def test_launcher_health_ready_when_all_lanes_are_stably_non_running() -> None:
    specs = {name: _spec(name) for name in ("baseline_parallel", "dividend", "paper_executor")}
    procs = {
        "baseline_parallel": DummyProc(101, 0),
        "dividend": DummyProc(102, 0),
    }

    payload = src._launcher_health_payload(
        specs=specs,
        procs=procs,  # type: ignore[arg-type]
        proc_started_at={"baseline_parallel": 1.0, "dividend": 1.0},
        restart_history={},
        quarantined_jobs={},
        launcher_started_at=1.0,
        phase="running",
        policy_parked_jobs={"paper_executor"},
        clean_exited_jobs={"baseline_parallel", "dividend"},
    )

    contract = payload["launcher_readiness_contract"]

    assert payload["overall_status"] == "ready"
    assert payload["running_job_count"] == 0
    assert payload["policy_parked_job_count"] == 1
    assert payload["clean_exited_job_count"] == 2
    assert payload["repair_packet"]["status"] == "clear"
    assert contract["readiness_status"] == "stable_with_parked_lanes"
    assert contract["class_counts"]["core_collection"]["stable_non_running"] == 2
    assert contract["class_counts"]["execution_lane"]["stable_non_running"] == 1


def test_quarantine_release_waits_for_cooldown_and_restart_budget() -> None:
    now = 10_000.0
    old_timestamp = datetime.fromtimestamp(now - 1_200, timezone.utc).isoformat()
    recent_timestamp = datetime.fromtimestamp(now - 60, timezone.utc).isoformat()

    restarts = [now - 3_700, now - 100]
    ready = src._quarantine_release_state(
        {"timestamp_utc": old_timestamp},
        restarts,
        now=now,
        cooldown_seconds=900,
        max_restarts_per_hour=2,
    )

    assert ready["ready"] is True
    assert ready["cooldown_ready"] is True
    assert ready["budget_ready"] is True
    assert ready["restart_count_last_hour"] == 1
    assert restarts == [now - 100]

    too_recent = src._quarantine_release_state(
        {"timestamp_utc": recent_timestamp},
        [],
        now=now,
        cooldown_seconds=900,
        max_restarts_per_hour=2,
    )
    assert too_recent["ready"] is False
    assert too_recent["cooldown_ready"] is False

    budget_full = src._quarantine_release_state(
        {"timestamp_utc": old_timestamp},
        [now - 100, now - 200],
        now=now,
        cooldown_seconds=900,
        max_restarts_per_hour=2,
    )
    assert budget_full["ready"] is False
    assert budget_full["budget_ready"] is False
