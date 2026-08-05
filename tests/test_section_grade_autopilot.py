import os
import sys
import time
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
    storage_call = next(cmd for cmd in calls if "storage-backpressure-autopilot" in cmd)
    assert "--quick-bounded" in storage_call
    assert payload["section_floor_autopilot_contract"]["bounded_step_timeouts"] is True
    assert payload["section_floor_autopilot_contract"]["bounded_storage_repairs"] is True
    assert payload["upgrade_track"]["upgradeable"] is True


def test_section_grade_autopilot_preserves_existing_quick_bounded_storage_command() -> None:
    command = [
        "./scripts/ops/opsctl.sh",
        "storage-backpressure-autopilot",
        "--apply",
        "--quick-bounded",
        "--json",
    ]

    assert src._bounded_repair_command(command) == command


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


def test_section_grade_autopilot_does_not_repair_managed_paper_advisories() -> None:
    guard = {
        "ok": True,
        "overall_status": "ready",
        "overall_letter_grade": "A+",
        "below_floor_count": 1,
        "blocking_below_floor_count": 0,
        "advisory_below_floor_count": 1,
        "below_floor_sections": ["live_trading_readiness"],
        "blocking_below_floor_sections": [],
        "advisory_below_floor_sections": ["live_trading_readiness"],
        "protected_by_floor_count": 0,
        "sections": [
            {
                "section": "live_trading_readiness",
                "state": "below_floor",
                "recommended_commands": [["./scripts/ops/opsctl.sh", "health"]],
            }
        ],
    }
    calls: list[list[str]] = []

    def runner(cmd: list[str], project_root: Path, timeout_sec: int) -> dict:
        calls.append(cmd)
        return {"cmd": cmd, "rc": 0, "payload": {}, "stdout_tail": "", "stderr_tail": ""}

    payload = src.build_payload(
        Path("/tmp/project"),
        apply=True,
        runner=runner,
        guard_builder=lambda _: dict(guard),
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["repair_step_count"] == 0
    assert payload["attempt_count"] == 0
    assert calls == []


def test_section_grade_autopilot_timeout_reaps_descendant_processes(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    parent = tmp_path / "spawn_child.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "open(sys.argv[1], 'w', encoding='utf-8').write(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )

    result = src._run([sys.executable, str(parent), str(child_pid_path)], tmp_path, 1)

    assert result["rc"] == 124
    assert result["timeout_cleanup"]["reaped"] is True
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    for _ in range(20):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError(f"timed-out maintenance child still alive: pid={child_pid}")
