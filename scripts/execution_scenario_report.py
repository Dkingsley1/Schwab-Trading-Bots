#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

from core.accountability import safe_write_json_atomic
from core.execution_scenarios import run_execution_scenarios


DEFAULT_OUT = Path("governance/health/execution_scenario_control_latest.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic broker and execution fault scenarios."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        **run_execution_scenarios(),
        "live_orders_enabled_by_this_control": False,
    }
    out = Path(args.out_file).expanduser()
    if not out.is_absolute():
        out = root / out
    safe_write_json_atomic(
        str(out), payload, project_root=str(root), source="execution_scenario_report"
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(
            f"execution_scenarios passed={payload['passed_count']}/{payload['scenario_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
