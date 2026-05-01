from pathlib import Path

from scripts.ops import section_grade_autopilot as src


def test_section_grade_autopilot_runs_floor_repairs_and_reports_final_guard() -> None:
    guard_payloads = [
        {
            "overall_status": "degraded",
            "overall_letter_grade": "A-",
            "below_floor_count": 0,
            "protected_by_floor_count": 2,
            "sections": [
                {
                    "section": "training_and_model_quality",
                    "state": "protected_by_floor",
                    "recommended_commands": [
                        ["./scripts/ops/opsctl.sh", "training-quality", "--json"],
                        ["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"],
                    ],
                },
                {
                    "section": "data_ingestion_and_storage",
                    "state": "protected_by_floor",
                    "recommended_commands": [
                        ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
                    ],
                },
            ],
            "recommended_actions": ["keep floor protected"],
        },
        {
            "overall_status": "ready",
            "overall_letter_grade": "A",
            "below_floor_count": 0,
            "protected_by_floor_count": 0,
            "sections": [],
            "recommended_actions": ["all sections at floor"],
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

    assert payload["overall_status"] == "ready"
    assert payload["attempt_count"] >= 4
    assert any("grade-regression-autopilot" in " ".join(cmd) for cmd in calls)
    assert any("training-quality" in " ".join(cmd) for cmd in calls)
    assert any("storage-backpressure-autopilot" in " ".join(cmd) for cmd in calls)
    assert payload["section_floor_autopilot_contract"]["bounded_step_timeouts"] is True
    assert payload["upgrade_track"]["upgradeable"] is True


def test_section_grade_autopilot_caps_nested_repair_timeouts() -> None:
    guard_payloads = [
        {
            "overall_status": "degraded",
            "overall_letter_grade": "A-",
            "below_floor_count": 0,
            "protected_by_floor_count": 1,
            "sections": [
                {
                    "section": "ops_and_autonomy",
                    "state": "protected_by_floor",
                    "recommended_commands": [["./scripts/ops/opsctl.sh", "autonomy-control", "--json"]],
                }
            ],
        },
        {
            "overall_status": "degraded",
            "overall_letter_grade": "A-",
            "below_floor_count": 0,
            "protected_by_floor_count": 1,
            "sections": [],
        },
    ]
    timeouts: list[int] = []

    def guard_builder(_: Path) -> dict:
        return guard_payloads.pop(0)

    def runner(cmd: list[str], project_root: Path, timeout_sec: int) -> dict:
        timeouts.append(timeout_sec)
        return {"cmd": cmd, "rc": 124, "payload": {}, "stdout_tail": "", "stderr_tail": "timeout"}

    payload = src.build_payload(
        Path("/tmp/project"),
        apply=True,
        max_step_timeout_sec=19,
        runner=runner,
        guard_builder=guard_builder,
    )

    assert payload["max_step_timeout_sec"] == 19
    assert timeouts
    assert all(timeout == 19 for timeout in timeouts)
    assert all(attempt["timeout_sec"] == 19 for attempt in payload["attempts"])
