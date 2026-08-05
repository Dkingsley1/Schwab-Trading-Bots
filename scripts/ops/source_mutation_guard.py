#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTECTED_PATHS = (
    ".github/workflows/ci_guardrails.yml",
    ".github/workflows/refresh-showcase.yml",
    "COMMANDS.md",
    "README.md",
    "core/master_bot.py",
    "scripts/install_production_hardening_watch_launchd.sh",
    "scripts/run_master_bot.py",
    "scripts/ops/command_validity_bot.py",
    "scripts/ops/commands_hygiene_bot.py",
    "scripts/ops/install_ops_automation_launchd.sh",
    "scripts/ops/use_mode_compliance_guard.py",
    "scripts/ops/commercial_readiness_control.py",
    "scripts/ops/live_canary_readiness_contract.py",
    "scripts/ops/paper_400_ramp_control.py",
    "scripts/ops/paper_live_data_standard.py",
    "scripts/ops/production_hardening_watch.py",
    "scripts/ops/run_production_hardening_watch_launchd.sh",
    "scripts/ops/infrabot_library_self_awareness_control.py",
    "scripts/ops/production_flow_smoke.py",
    "scripts/ops/production_level_upgrade_hardener_control.py",
    "scripts/ops/production_quality_control.py",
    "scripts/ops/production_excellence_control.py",
    "scripts/ops/live_order_ledger_control.py",
    "scripts/ops/production_quality_slo_guard.py",
    "scripts/ops/production_readiness_control.py",
    "scripts/ops/runtime_throttle_control.py",
    "scripts/ops/source_mutation_guard.py",
    "config/deployment_profiles.json",
    "config/self_healing_policy.json",
    "config/credential_runtime_policy.json",
    "config/promotion_gate_snapshot_policy.json",
    "config/generated_artifact_policy.json",
    "config/use_mode_compliance_policy_v1.json",
    "config/commercial_readiness_framework_v1.json",
    "config/infrabot_library_self_awareness_v1.json",
    "config/live_canary_readiness_contract.json",
    "config/production_level_upgrade_hardener_v1.json",
    "config/production_readiness_control_v1.json",
    "config/production_excellence_v1.json",
    "core/live_order_ledger.py",
    "docs/operations/PRODUCTION_FLOW_GUARDRAILS.md",
    "docs/operations/USE_MODE_COMPLIANCE_GUARDRAILS.md",
    "docs/operations/COMMERCIAL_READINESS_FRAMEWORK.md",
    "master_bot_registry.json",
)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_status(project_root: Path, protected_paths: tuple[str, ...] = DEFAULT_PROTECTED_PATHS) -> tuple[list[str], str]:
    cmd = ["git", "status", "--porcelain", "--", *protected_paths]
    proc = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return [], (proc.stderr or proc.stdout or "git status failed").strip()
    return [line for line in proc.stdout.splitlines() if line.strip()], ""


def build_payload(project_root: Path, protected_paths: tuple[str, ...] = DEFAULT_PROTECTED_PATHS) -> dict[str, Any]:
    dirty_entries, error = git_status(project_root, protected_paths=protected_paths)
    ok = not dirty_entries and not error
    return {
        "timestamp_utc": iso_now(),
        "ok": ok,
        "overall_status": "ready" if ok else "blocked",
        "check": "source_mutation_guard",
        "project_root": str(project_root),
        "protected_paths": list(protected_paths),
        "dirty_count": len(dirty_entries),
        "dirty_entries": dirty_entries,
        "error": error,
        "contract": {
            "runtime_outputs_only": ["governance", "runtime", "exports", "logs", "tmp"],
            "canonical_source_updates_require_explicit_operator_intent": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if runtime or CI has mutated protected source files.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--check-clean", action="store_true", help="Exit nonzero when protected source paths are dirty.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    payload = build_payload(Path(args.project_root).resolve())
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "source_mutation_guard "
            f"status={payload['overall_status']} "
            f"dirty_count={payload['dirty_count']}"
        )
        for entry in payload["dirty_entries"]:
            print(entry)
        if payload["error"]:
            print(payload["error"], file=sys.stderr)

    if args.check_clean and not payload["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
