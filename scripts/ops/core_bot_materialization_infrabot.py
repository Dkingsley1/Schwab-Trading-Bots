#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import core_bot_materialization_guard as guard
import materialize_core_bot_modules as materializer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "core_bot_materialization_infrabot_latest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_catalog(project_root: Path) -> dict[str, Any]:
    cmd = [
        str(project_root / ".venv312" / "bin" / "python"),
        str(project_root / "scripts" / "ops" / "build_core_bot_catalog.py"),
        "--json",
    ]
    if not Path(cmd[0]).exists():
        cmd[0] = "python3"
    result = subprocess.run(cmd, cwd=str(project_root), text=True, capture_output=True, check=False)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip()[-2000:],
        "stderr": result.stderr.strip()[-2000:],
    }


def _run_tier_organizer(project_root: Path) -> dict[str, Any]:
    cmd = [
        str(project_root / ".venv312" / "bin" / "python"),
        str(project_root / "scripts" / "ops" / "organize_core_bot_tiers.py"),
        "--json",
    ]
    if not Path(cmd[0]).exists():
        cmd[0] = "python3"
    result = subprocess.run(cmd, cwd=str(project_root), text=True, capture_output=True, check=False)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip()[-2000:],
        "stderr": result.stderr.strip()[-2000:],
    }


def run(project_root: Path = PROJECT_ROOT, apply: bool = False, overwrite_generated: bool = False) -> dict[str, Any]:
    before = guard.build_payload(project_root)
    materialize_payload: dict[str, Any] | None = None
    catalog_payload: dict[str, Any] | None = None
    if apply or before["overall_status"] != "ready":
        materialize_payload = materializer.materialize(project_root, overwrite_generated=overwrite_generated)
        catalog_payload = _run_catalog(project_root)
    tier_payload = _run_tier_organizer(project_root)
    after = guard.build_payload(project_root)
    return {
        "overall_status": after["overall_status"],
        "generated_at_utc": _utc_now(),
        "apply": bool(apply),
        "before": before,
        "materialize": materialize_payload,
        "catalog_refresh": catalog_payload,
        "tier_refresh": tier_payload,
        "after": after,
        "recommended_actions": after.get("recommended_actions", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Infrastructure bot that keeps expansion bots visible as core/*.py modules.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite-generated", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    payload = run(project_root, apply=bool(args.apply), overwrite_generated=bool(args.overwrite_generated))
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        summary = payload["after"]["summary"]
        print(
            "core_bot_materialization_infrabot "
            f"status={payload['overall_status']} "
            f"present={summary['present_core_module_count']} "
            f"missing={summary['missing_core_module_count']}"
        )
    return 0 if payload["overall_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
