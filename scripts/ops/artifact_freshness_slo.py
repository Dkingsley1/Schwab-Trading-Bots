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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "artifact_freshness_slo_latest.json"


def _artifact_contract(project_root: Path) -> dict[str, dict[str, Any]]:
    return {
        "session_ready": {
            "path": project_root / "governance" / "health" / "session_ready_latest.json",
            "max_age_minutes": 15.0,
            "required": True,
            "refresh_command": "./scripts/session_ready_check.py --json",
        },
        "process_watchdog": {
            "path": project_root / "governance" / "health" / "process_watchdog_latest.json",
            "max_age_minutes": 45.0,
            "required": True,
            "refresh_command": "./scripts/ops/opsctl.sh status",
        },
        "live_readiness_smoke": {
            "path": project_root / "governance" / "health" / "live_readiness_smoke_latest.json",
            "max_age_minutes": 240.0,
            "required": True,
            "refresh_command": "./scripts/live_readiness_smoke.py --json",
        },
        "runtime_training_snapshot": {
            "path": project_root / "governance" / "health" / "runtime_training_snapshot_latest.json",
            "max_age_minutes": 24.0 * 60.0,
            "required": False,
            "refresh_command": "./scripts/ops/opsctl.sh runtime-training-snapshot --json",
        },
        "storage_resilience_control": {
            "path": project_root / "governance" / "health" / "storage_resilience_control_latest.json",
            "max_age_minutes": 48.0 * 60.0,
            "required": False,
            "refresh_command": "./scripts/ops/opsctl.sh storage-resilience --json",
        },
        "operator_cockpit": {
            "path": project_root / "governance" / "health" / "operator_cockpit_latest.json",
            "max_age_minutes": 24.0 * 60.0,
            "required": False,
            "refresh_command": "./scripts/ops/opsctl.sh operator-cockpit --json",
        },
        "sentiment_report": {
            "path": project_root / "governance" / "health" / "sentiment_report_latest.json",
            "max_age_minutes": 48.0 * 60.0,
            "required": False,
            "refresh_command": "./scripts/ops/opsctl.sh sentiment-report --json",
        },
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    breaches: list[dict[str, Any]] = []
    contract = _artifact_contract(project_root)
    stale_required = 0
    stale_optional = 0
    for name, cfg in contract.items():
        path = Path(cfg["path"])
        payload = load_json(path)
        exists = path.exists() and bool(payload)
        age_minutes = payload_age_minutes(payload, path) if exists else None
        stale = not exists or (age_minutes is not None and float(age_minutes) > float(cfg["max_age_minutes"]))
        if stale:
            if bool(cfg["required"]):
                stale_required += 1
            else:
                stale_optional += 1
        breaches.append(
            {
                "name": name,
                "path": str(path),
                "required": bool(cfg["required"]),
                "exists": exists,
                "age_minutes": round(float(age_minutes), 4) if age_minutes is not None else None,
                "max_age_minutes": float(cfg["max_age_minutes"]),
                "stale": bool(stale),
                "refresh_command": str(cfg["refresh_command"]),
            }
        )

    overall_status = "ready"
    if stale_required > 0:
        overall_status = "blocked"
    elif stale_optional > 1:
        overall_status = "degraded"

    recommended_actions = ordered_unique([row["refresh_command"] for row in breaches if bool(row.get("stale"))])[:8]

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "sla_summary": {
            "artifact_count": len(breaches),
            "stale_required": stale_required,
            "stale_optional": stale_optional,
        },
        "artifacts": breaches,
        "infra_bots": ["artifact_freshness_slo", "retrain_artifact_freshness_guard", "runtime_gate_dashboard"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce freshness SLAs across the core runtime artifacts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "artifact_freshness_slo "
            f"overall_status={payload.get('overall_status', '')} "
            f"stale_required={int(((payload.get('sla_summary') or {}).get('stale_required', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
