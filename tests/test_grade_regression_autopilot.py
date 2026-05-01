from pathlib import Path

from scripts.ops import grade_regression_autopilot as src


def test_grade_regression_autopilot_runs_targeted_repairs_and_reports_final_guard() -> None:
    guard_payloads = [
        {
            "overall_status": "blocked",
            "blocked_surface_count": 2,
            "degraded_surface_count": 3,
            "surfaces": [
                {"surface": "training_lineage", "state": "degraded", "retry_budget": {"step_timeout_sec": 33, "max_attempts_per_run": 2}},
                {"surface": "storage_control", "state": "degraded", "retry_budget": {"step_timeout_sec": 44, "quiet_hours_preferred": True}},
                {"surface": "incident_closeout", "state": "degraded", "retry_budget": {"step_timeout_sec": 55}},
            ],
            "recommended_actions": ["lift training lineage", "drain storage"],
        },
        {
            "overall_status": "degraded",
            "blocked_surface_count": 0,
            "degraded_surface_count": 2,
            "surfaces": [],
            "recommended_actions": ["keep watching"],
        },
    ]
    calls: list[list[str]] = []

    def guard_builder(_: Path) -> dict:
        return guard_payloads.pop(0)

    def runner(cmd: list[str], project_root: Path, timeout_sec: int) -> dict:
        calls.append(list(cmd))
        return {"cmd": cmd, "rc": 0, "payload": {"overall_status": "ready"}, "stdout_tail": "", "stderr_tail": ""}

    payload = src.build_payload(
        Path("/tmp/project"),
        apply=True,
        runner=runner,
        guard_builder=guard_builder,
    )

    assert payload["overall_status"] == "degraded"
    assert payload["attempt_count"] >= 4
    assert any("training_lineage_manifest.py" in " ".join(cmd) for cmd in calls)
    assert any("storage_backpressure_autopilot.py" in " ".join(cmd) for cmd in calls)
    assert any("incident_closeout_autopilot.py" in " ".join(cmd) for cmd in calls)
    assert payload["regression_autopilot_contract"]["uses_per_surface_retry_budgets"] is True
    assert any(step["quiet_hours_preferred"] for step in payload["repair_plan"])
    assert payload["upgrade_track"]["upgradeable"] is True
