from __future__ import annotations

import sys
import time
from pathlib import Path

from scripts.ops import system_drift_autopilot as src


def test_system_drift_autopilot_repairs_blocked_surfaces(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    guards = [
        {
            "overall_status": "blocked",
            "metrics": {"blocked_surface_count": 1, "degraded_surface_count": 0},
            "surfaces": [
                {
                    "name": "command_validity",
                    "family": "command_surface",
                    "status": "blocked",
                    "repair_commands": [["./scripts/ops/opsctl.sh", "command-validity", "--apply", "--json"]],
                }
            ],
        },
        {
            "overall_status": "ready",
            "metrics": {"blocked_surface_count": 0, "degraded_surface_count": 0},
            "surfaces": [
                {
                    "name": "command_validity",
                    "family": "command_surface",
                    "status": "ready",
                    "repair_commands": [["./scripts/ops/opsctl.sh", "command-validity", "--apply", "--json"]],
                }
            ],
        },
    ]

    def guard_builder(_project_root: Path) -> dict:
        return guards.pop(0)

    def runner(cmd: list[str], _project_root: Path, _timeout_sec: int) -> dict:
        calls.append(list(cmd))
        return {"cmd": list(cmd), "rc": 0, "payload": {"overall_status": "ready"}, "stdout_tail": "", "stderr_tail": ""}

    payload = src.build_payload(tmp_path, apply=True, guard_builder=guard_builder, runner=runner)

    assert payload["overall_status"] == "ready"
    assert payload["repair_step_count"] == 1
    assert payload["attempt_count"] == 1
    assert calls == [["./scripts/ops/opsctl.sh", "command-validity", "--apply", "--json"]]


def test_system_drift_autopilot_surfaces_operator_followup_when_no_safe_repair_exists(tmp_path: Path) -> None:
    def guard_builder(_project_root: Path) -> dict:
        return {
            "overall_status": "blocked",
            "metrics": {"blocked_surface_count": 1, "degraded_surface_count": 0},
            "surfaces": [
                {
                    "name": "infrastructure_autofix",
                    "family": "infrastructure_surface",
                    "status": "blocked",
                    "repair_commands": [],
                }
            ],
        }

    payload = src.build_payload(tmp_path, apply=False, guard_builder=guard_builder)

    assert payload["overall_status"] == "blocked"
    assert payload["operator_followups"] == ["infrastructure_autofix"]
    assert payload["repair_step_count"] == 0


def test_system_drift_autopilot_skips_reporting_repairs_when_chrome_guard_is_blocked(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def guard_builder(_project_root: Path) -> dict:
        return {
            "overall_status": "blocked",
            "metrics": {"blocked_surface_count": 2, "degraded_surface_count": 0},
            "surfaces": [
                {
                    "name": "chrome_headless_guard",
                    "family": "workstation_surface",
                    "status": "blocked",
                    "repair_commands": [["./scripts/ops/opsctl.sh", "chrome-headless-guard", "--apply", "--json"]],
                },
                {
                    "name": "report_pdf_bundle",
                    "family": "reporting_surface",
                    "status": "blocked",
                    "repair_commands": [["./scripts/ops/opsctl.sh", "report-pdfs", "--json"]],
                },
            ],
        }

    def runner(cmd: list[str], _project_root: Path, _timeout_sec: int) -> dict:
        calls.append(list(cmd))
        return {"cmd": list(cmd), "rc": 0, "payload": {"overall_status": "ready"}, "stdout_tail": "", "stderr_tail": ""}

    payload = src.build_payload(tmp_path, apply=True, guard_builder=guard_builder, runner=runner)

    assert calls == [["./scripts/ops/opsctl.sh", "chrome-headless-guard", "--apply", "--json"]]
    assert payload["planned_repair_step_count"] == 2
    assert payload["repair_step_count"] == 1
    assert payload["skipped_step_count"] == 1
    assert payload["skipped_steps"][0]["surface"] == "report_pdf_bundle"
    assert payload["skipped_steps"][0]["skip_reason"] == "chrome_guard_not_ready"


def test_system_drift_autopilot_caps_nested_repair_timeouts(tmp_path: Path) -> None:
    timeouts: list[int] = []
    guards = [
        {
            "overall_status": "blocked",
            "metrics": {"blocked_surface_count": 1, "degraded_surface_count": 0},
            "surfaces": [
                {
                    "name": "architecture_upgrade_scoreboard",
                    "family": "architecture_surface",
                    "status": "blocked",
                    "repair_commands": [["./scripts/ops/opsctl.sh", "slow-repair", "--json"]],
                }
            ],
        },
        {
            "overall_status": "degraded",
            "metrics": {"blocked_surface_count": 0, "degraded_surface_count": 1},
            "surfaces": [],
        },
    ]

    def guard_builder(_project_root: Path) -> dict:
        return guards.pop(0)

    def runner(cmd: list[str], _project_root: Path, timeout_sec: int) -> dict:
        timeouts.append(timeout_sec)
        return {"cmd": list(cmd), "rc": 124, "payload": {}, "stdout_tail": "", "stderr_tail": "timeout"}

    payload = src.build_payload(
        tmp_path,
        apply=True,
        max_step_timeout_sec=17,
        guard_builder=guard_builder,
        runner=runner,
    )

    assert payload["max_step_timeout_sec"] == 17
    assert timeouts == [17]
    assert payload["attempts"][0]["timeout_sec"] == 17


def test_system_drift_autopilot_skips_recovery_deferred_surfaces(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    guards = [
        {
            "overall_status": "degraded",
            "metrics": {"blocked_surface_count": 0, "degraded_surface_count": 2},
            "surfaces": [
                {
                    "name": "adaptive_regression_guard",
                    "family": "governance_surface",
                    "status": "degraded",
                    "recovery_deferred": True,
                    "recovery_deferred_reason": "pressure_deferred_count=2",
                    "repair_commands": [["./scripts/ops/opsctl.sh", "adaptive-regression-guard", "--apply", "--json"]],
                },
                {
                    "name": "commands_hygiene",
                    "family": "command_surface",
                    "status": "degraded",
                    "repair_commands": [["./scripts/ops/opsctl.sh", "commands-hygiene", "--apply", "--json"]],
                },
            ],
        },
        {
            "overall_status": "degraded",
            "metrics": {"blocked_surface_count": 0, "degraded_surface_count": 1},
            "surfaces": [],
        },
    ]

    def guard_builder(_project_root: Path) -> dict:
        return guards.pop(0)

    def runner(cmd: list[str], _project_root: Path, _timeout_sec: int) -> dict:
        calls.append(list(cmd))
        return {"cmd": list(cmd), "rc": 0, "payload": {"overall_status": "ready"}, "stdout_tail": "", "stderr_tail": ""}

    payload = src.build_payload(tmp_path, apply=True, guard_builder=guard_builder, runner=runner)

    assert calls == [["./scripts/ops/opsctl.sh", "commands-hygiene", "--apply", "--json"]]
    assert payload["repair_step_count"] == 1
    assert payload["skipped_step_count"] == 1
    assert payload["skipped_steps"][0]["skip_reason"] == "recovery_deferred"


def test_system_drift_autopilot_run_timeout_returns_clean_failure(tmp_path: Path) -> None:
    result = src._run([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, 1)

    assert result["rc"] == 124
    assert result["stderr_tail"] == "timeout"


def test_system_drift_autopilot_timeout_reaps_known_one_numbers_child(tmp_path: Path) -> None:
    child = tmp_path / "scripts" / "build_one_numbers_report.py"
    child.parent.mkdir(parents=True, exist_ok=True)
    child.write_text("import time\ntime.sleep(20)\n", encoding="utf-8")
    parent = tmp_path / "spawn_child.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, sys.argv[1]], start_new_session=True)\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )

    result = src._run([sys.executable, str(parent), str(child), "one-numbers-regression-guard"], tmp_path, 1)

    assert result["rc"] == 124
    assert result["timeout_cleanup"]["terminated_processes"]
    for _ in range(20):
        if not src._project_processes(tmp_path, ["scripts/build_one_numbers_report.py"]):
            break
        time.sleep(0.1)
    assert src._project_processes(tmp_path, ["scripts/build_one_numbers_report.py"]) == {}
