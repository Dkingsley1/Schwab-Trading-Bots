from __future__ import annotations

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
