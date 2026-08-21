#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
    contract = {
        "bot_organization_control": {
            "path": project_root / "governance" / "health" / "bot_organization_latest.json",
            "max_age_minutes": 24.0 * 60.0,
            "required": True,
            "refresh_command": "./scripts/ops/opsctl.sh bot-organization --json",
        },
        "bot_profitability_scalability_control": {
            "path": project_root
            / "governance"
            / "health"
            / "bot_profitability_scalability_latest.json",
            "max_age_minutes": 120.0,
            "required": True,
            "refresh_command": "./scripts/ops/opsctl.sh bot-profitability-scalability --json",
        },
        "master_grandmaster_evidence_v2": {
            "path": project_root / "governance" / "health" / "master_grandmaster_evidence_v2_latest.json",
            "max_age_minutes": 120.0,
            "required": True,
            "refresh_command": "./scripts/ops/opsctl.sh master-grandmaster-evidence --json",
        },
        "control_surface_ownership": {
            "path": project_root / "governance" / "health" / "control_surface_ownership_latest.json",
            "max_age_minutes": 60.0,
            "required": True,
            "refresh_command": "./scripts/ops/opsctl.sh control-surface-ownership --json",
        },
        "system_role_contract": {
            "path": project_root / "governance" / "health" / "system_role_contract_latest.json",
            "max_age_minutes": 30.0,
            "required": True,
            "refresh_command": "./scripts/ops/opsctl.sh system-role-contract --json",
        },
        "independent_runtime_monitor": {
            "path": project_root / "governance" / "health" / "independent_runtime_monitor_latest.json",
            "max_age_minutes": 10.0,
            "required": True,
            "refresh_command": "./scripts/ops/opsctl.sh independent-runtime-monitor --json",
        },
        "production_resilience_control": {
            "path": project_root / "governance" / "health" / "production_resilience_control_latest.json",
            "max_age_minutes": 30.0,
            "required": True,
            "refresh_command": "./scripts/ops/opsctl.sh production-resilience --json",
        },
        "soak_reliability_sentinel": {
            "path": project_root / "governance" / "health" / "soak_reliability_sentinel_latest.json",
            "max_age_minutes": 20.0,
            "required": True,
            "refresh_command": "./scripts/ops/soak_reliability_sentinel.py --apply --json",
        },
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
    if (project_root / "config" / "collector_capability_catalog_v1.json").is_file():
        contract["capability_materialization"] = {
            "path": project_root
            / "governance"
            / "collector_capabilities"
            / "materialized_capabilities_latest.json",
            "max_age_minutes": 30.0,
            "required": True,
            "refresh_command": "./scripts/ops/opsctl.sh capability-materialization --json",
        }
        contract["collector_capability_control"] = {
            "path": project_root / "governance" / "health" / "collector_capability_control_latest.json",
            "max_age_minutes": 30.0,
            "required": True,
            "refresh_command": "./scripts/ops/opsctl.sh collector-capability-control --json",
        }
    return contract


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
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if exists else "",
            }
        )

    overall_status = "ready"
    if stale_required > 0:
        overall_status = "blocked"
    elif stale_optional > 1:
        overall_status = "degraded"

    recommended_actions = ordered_unique([row["refresh_command"] for row in breaches if bool(row.get("stale"))])[:8]

    receipt_rows = {
        row["name"]: {
            "source_sha256": row["source_sha256"],
            "age_minutes": row["age_minutes"],
            "stale": row["stale"],
        }
        for row in breaches
    }
    receipt_sha = hashlib.sha256(
        json.dumps(receipt_rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
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
        "evidence_epoch": {
            "id": f"artifact-freshness:{receipt_sha[:16]}",
            "receipt_sha256": receipt_sha,
            "source_count": len(breaches),
        },
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
