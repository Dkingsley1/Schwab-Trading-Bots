#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_drift_registry_latest.json"


def _python_bin(project_root: Path) -> str:
    preferred = project_root / ".venv314" / "bin" / "python"
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
            "name": "adaptive_regression_guard",
            "family": "governance_surface",
            "artifact_path": health_root / "adaptive_regression_guard_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 30,
            "repair_commands": [
                ["./scripts/ops/opsctl.sh", "adaptive-regression-guard", "--apply", "--json"],
                ["./scripts/ops/opsctl.sh", "grade-regression-autopilot", "--apply", "--json"],
                ["./scripts/ops/opsctl.sh", "section-grade-autopilot", "--apply", "--json"],
            ],
            "notes": [
                "Learns persistence across grade, section, and runtime regression guards before escalating repairs.",
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
            "name": "paper_execution_truth_layer",
            "family": "paper_trading_surface",
            "artifact_path": health_root / "paper_execution_truth_layer_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 180,
            "repair_commands": [["./scripts/ops/opsctl.sh", "paper-truth", "--json"]],
            "assigned_bot": "paper_execution_truth_layer",
            "owner_bot": "infrabot_adaptive_governor",
            "notes": [
                "Watch with ok=true and no failed checks is managed attribution debt, not an execution blocker.",
            ],
        },
        {
            "name": "paper_profitability_control",
            "family": "paper_trading_surface",
            "artifact_path": health_root / "paper_profitability_control_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 180,
            "repair_commands": [["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"]],
            "assigned_bot": "paper_profitability_control",
            "owner_bot": "infrabot_adaptive_governor",
            "notes": [
                "Visible raw D/F evidence remains tracked, but active_blocker_count=0 keeps paper profitability from blocking drift.",
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
            "name": "system_architecture_contract_graph",
            "family": "architecture_surface",
            "artifact_path": health_root / "system_architecture_contract_graph_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 30,
            "repair_commands": [
                ["./scripts/ops/opsctl.sh", "system-architecture-contract-graph", "--apply", "--json"],
                ["./scripts/ops/opsctl.sh", "adaptive-regression-guard", "--apply", "--json"],
                ["./scripts/ops/opsctl.sh", "distributed-cell-architecture", "--apply", "--json"],
            ],
            "notes": [
                "Maps architecture artifacts, dependencies, freshness, and authority boundaries into a system-wide contract graph.",
            ],
        },
        {
            "name": "system_architecture_autopilot",
            "family": "architecture_surface",
            "artifact_path": health_root / "system_architecture_autopilot_latest.json",
            "status_key": "overall_status",
            "ok_key": "ok",
            "max_age_minutes": 30,
            "repair_commands": [
                ["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"],
                ["./scripts/ops/opsctl.sh", "system-architecture-contract-graph", "--apply", "--json"],
            ],
            "notes": [
                "Plans dependency-ordered architecture repairs from the contract graph; command execution requires explicit --execute-safe-repairs.",
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


def _jsonable_command(command: Any) -> list[str]:
    if not isinstance(command, list):
        return []
    return [str(part) for part in command]


def _jsonable_spec(spec: dict[str, Any]) -> dict[str, Any]:
    repair_commands = [
        command
        for command in (_jsonable_command(raw) for raw in list(spec.get("repair_commands") or []))
        if command
    ]
    return {
        "name": str(spec.get("name") or ""),
        "family": str(spec.get("family") or ""),
        "artifact_path": str(spec.get("artifact_path") or ""),
        "kind": str(spec.get("kind") or ""),
        "status_key": str(spec.get("status_key") or ""),
        "ok_key": str(spec.get("ok_key") or ""),
        "max_age_minutes": spec.get("max_age_minutes"),
        "repair_commands": repair_commands,
        "repairable": bool(repair_commands),
        "assigned_bot": str(spec.get("assigned_bot") or ""),
        "owner_bot": str(spec.get("owner_bot") or ""),
        "notes": [str(note) for note in list(spec.get("notes") or [])],
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    specs = [_jsonable_spec(spec) for spec in surface_specs(project_root)]
    family_counts: dict[str, int] = {}
    for spec in specs:
        family = str(spec.get("family") or "other")
        family_counts[family] = family_counts.get(family, 0) + 1

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready",
        "surface_count": len(specs),
        "repairable_surface_count": sum(1 for spec in specs if spec.get("repairable")),
        "family_counts": family_counts,
        "surfaces": specs,
        "recommended_commands": [["./scripts/ops/opsctl.sh", "system-drift-guard", "--json"]],
        "policy": "registry_declares_drift_surfaces; guard_evaluates_current_artifact_state",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit the system drift surface registry artifact.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_file = Path(args.out_file).expanduser()
    payload = build_payload(project_root)
    write_payload(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_drift_registry "
            f"overall_status={payload.get('overall_status', '')} "
            f"surfaces={int(payload.get('surface_count', 0) or 0)} "
            f"repairable={int(payload.get('repairable_surface_count', 0) or 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
