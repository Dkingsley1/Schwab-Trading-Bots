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


def test_grade_regression_autopilot_omits_recursive_refresh_inside_refresh_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RUNTIME_ARTIFACT_REFRESH_ACTIVE", "1")

    plan = src._repair_plan(
        tmp_path,
        {
            "surfaces": [
                {
                    "surface": "storage_control",
                    "state": "degraded",
                    "retry_budget": {"step_timeout_sec": 30},
                }
            ]
        },
        storage_max_cycles=1,
    )

    assert not any("runtime_artifact_refresh.py" in " ".join(step.get("cmd") or []) for step in plan)
    assert any("ingestion_storage_control.py" in " ".join(step.get("cmd") or []) for step in plan)
    assert not any("storage_backpressure_autopilot.py" in " ".join(step.get("cmd") or []) for step in plan)


def test_grade_regression_autopilot_is_noop_when_every_surface_is_ready(tmp_path: Path) -> None:
    plan = src._repair_plan(
        tmp_path,
        {
            "overall_status": "ready",
            "surfaces": [
                {"surface": "training_quality", "state": "ready"},
                {"surface": "storage_control", "state": "ready"},
            ],
        },
        storage_max_cycles=1,
    )

    assert plan == []


def test_grade_regression_autopilot_never_embeds_full_artifact_refresh(tmp_path: Path) -> None:
    plan = src._repair_plan(
        tmp_path,
        {
            "overall_status": "blocked",
            "surfaces": [
                {"surface": "training_quality", "state": "blocked"},
                {"surface": "storage_control", "state": "degraded"},
            ],
        },
        storage_max_cycles=1,
    )

    assert plan
    assert not any("runtime_artifact_refresh.py" in " ".join(step.get("cmd") or []) for step in plan)
