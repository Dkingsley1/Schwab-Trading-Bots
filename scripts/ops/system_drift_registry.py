#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _python_bin(project_root: Path) -> str:
    preferred = project_root / ".venv312" / "bin" / "python"
    if preferred.exists():
        return str(preferred)
    return sys.executable


def _py_cmd(project_root: Path, relative_script: str, *args: str) -> list[str]:
    return [_python_bin(project_root), str(project_root / relative_script), *args]


def surface_specs(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    health_root = project_root / "governance" / "health"
    return [
        {
            "name": "commands_hygiene",
            "family": "command_surface",
            "artifact_path": health_root / "commands_hygiene_latest.json",
            "kind": "commands_hygiene",
            "max_age_minutes": 30,
            "repair_commands": [["./scripts/ops/opsctl.sh", "commands-hygiene", "--apply", "--json"]],
            "notes": [
                "Keeps COMMANDS.md and the runbook script aligned with the curated operator inventory.",
            ],
        },
        {
            "name": "command_validity",
            "family": "command_surface",
            "artifact_path": health_root / "command_validity_latest.json",
            "kind": "command_validity",
            "max_age_minutes": 30,
            "repair_commands": [
                ["./scripts/ops/opsctl.sh", "commands-hygiene", "--apply", "--json"],
                ["./scripts/ops/opsctl.sh", "command-validity", "--apply", "--json"],
            ],
            "notes": [
                "Treats operator-gated commands as expected while still failing on broken safe probes.",
            ],
        },
        {
            "name": "codex_project_guard",
            "family": "command_surface",
            "artifact_path": health_root / "codex_project_guard_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 30,
            "repair_commands": [["./scripts/ops/opsctl.sh", "codex-project-guard", "--staged", "--json"]],
            "notes": [
                "Keeps Codex-authored work anchored to AGENTS.md and the system source-of-truth map.",
            ],
        },
        {
            "name": "section_grade_guard",
            "family": "governance_surface",
            "artifact_path": health_root / "section_grade_guard_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 45,
            "repair_commands": [
                ["./scripts/ops/opsctl.sh", "section-grade-guard", "--json"],
                ["./scripts/ops/opsctl.sh", "section-grade-autopilot", "--apply", "--json"],
            ],
        },
        {
            "name": "grade_regression_guard",
            "family": "governance_surface",
            "artifact_path": health_root / "grade_regression_guard_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 45,
            "repair_commands": [
                ["./scripts/ops/opsctl.sh", "grade-regression-guard", "--json"],
                ["./scripts/ops/opsctl.sh", "grade-regression-autopilot", "--apply", "--json"],
            ],
        },
        {
            "name": "one_numbers_regression_guard",
            "family": "analytics_surface",
            "artifact_path": health_root / "one_numbers_regression_guard_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 30,
            "repair_commands": [["./scripts/ops/opsctl.sh", "one-numbers-regression-guard", "--apply", "--json"]],
            "assigned_bot": "system_drift_autopilot",
            "owner_bot": "infrastructure_autofix_bot",
            "notes": [
                "One Numbers is full-rebuild only; drift repair must not regenerate lightweight cached CSV output.",
            ],
        },
        {
            "name": "chrome_headless_guard",
            "family": "workstation_surface",
            "artifact_path": health_root / "chrome_headless_guard_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 20,
            "repair_commands": [["./scripts/ops/opsctl.sh", "chrome-headless-guard", "--apply", "--json"]],
        },
        {
            "name": "system_summary_report",
            "family": "reporting_surface",
            "artifact_path": health_root / "system_summary_report_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 120,
            "repair_commands": [["./scripts/ops/opsctl.sh", "system-summary", "--refresh-supporting-artifacts", "--render-pdf", "--json"]],
        },
        {
            "name": "system_summary_autopilot",
            "family": "reporting_surface",
            "artifact_path": health_root / "system_summary_autopilot_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 120,
            "repair_commands": [_py_cmd(project_root, "scripts/ops/system_summary_autopilot.py", "--json")],
        },
        {
            "name": "report_pdf_bundle",
            "family": "reporting_surface",
            "artifact_path": health_root / "report_pdf_bundle_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 120,
            "repair_commands": [["./scripts/ops/opsctl.sh", "report-pdfs", "--json"]],
        },
        {
            "name": "architecture_upgrade_scoreboard",
            "family": "architecture_surface",
            "artifact_path": health_root / "architecture_upgrade_scoreboard_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 90,
            "repair_commands": [
                _py_cmd(project_root, "scripts/ops/mode_switchboard_mission_control.py", "--json"),
                _py_cmd(project_root, "scripts/ops/decision_provenance_cards.py", "--json"),
                _py_cmd(project_root, "scripts/ops/autonomy_control_plane.py", "--json"),
                _py_cmd(project_root, "scripts/ops/architecture_upgrade_scoreboard.py", "--json"),
            ],
        },
        {
            "name": "process_watchdog",
            "family": "stack_runtime",
            "artifact_path": health_root / "process_watchdog_latest.json",
            "kind": "watchdog",
            "max_age_minutes": 15,
            "repair_commands": [_py_cmd(project_root, "scripts/ops/process_watchdog.py", "--json")],
        },
        {
            "name": "incident_closeout",
            "family": "safety_surface",
            "artifact_path": health_root / "incident_closeout_autopilot_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 45,
            "repair_commands": [["./scripts/ops/opsctl.sh", "incident-closeout", "--json"]],
        },
        {
            "name": "live_runtime_separation",
            "family": "safety_surface",
            "artifact_path": health_root / "live_runtime_separation_control_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 45,
            "repair_commands": [["./scripts/ops/opsctl.sh", "live-runtime-separation", "--json"]],
        },
        {
            "name": "infrastructure_autofix",
            "family": "infrastructure_surface",
            "artifact_path": health_root / "infrastructure_autofix_bot_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 45,
            "repair_commands": [],
            "notes": [
                "This surface is repaired by its own launchd cadence and by the drift autopilot parent orchestration.",
            ],
        },
        {
            "name": "master_infrastructure_supervisor",
            "family": "infrastructure_surface",
            "artifact_path": health_root / "master_infrastructure_supervisor_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 30,
            "repair_commands": [["./scripts/ops/opsctl.sh", "master-infra-supervisor", "--json"]],
            "notes": [
                "Parent infrastructure supervisor that verifies child bots, storage routes, command docs, report jobs, and One Numbers original-start coverage together.",
            ],
        },
        {
            "name": "coinbase_api_health",
            "family": "broker_surface",
            "artifact_path": health_root / "coinbase_api_health_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 30,
            "repair_commands": [["./scripts/ops/opsctl.sh", "coinbase-api-health", "--json"]],
            "assigned_bot": "master_infrastructure_supervisor",
            "owner_bot": "infrastructure_autofix_bot",
            "notes": [
                "Checks Coinbase public market-data API availability without exposing credential values.",
            ],
        },
        {
            "name": "point_in_time_event_store",
            "family": "replay_surface",
            "artifact_path": health_root / "point_in_time_event_store_latest.json",
            "ok_key": "ok",
            "max_age_minutes": 60,
            "repair_commands": [["./scripts/ops/opsctl.sh", "point-in-time-event-store", "--json"]],
        },
        {
            "name": "replay_hash_registry_guard",
            "family": "replay_surface",
            "artifact_path": health_root / "replay_hash_registry_guard_latest.json",
            "ok_key": "ok",
            "max_age_minutes": 60,
            "repair_commands": [["./scripts/ops/opsctl.sh", "replay-hash-registry", "--json"]],
        },
        {
            "name": "golden_replay_regression_guard",
            "family": "replay_surface",
            "artifact_path": health_root / "golden_replay_regression_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 60,
            "repair_commands": [["./scripts/ops/opsctl.sh", "golden-replay-regression", "--json"]],
        },
    ]
