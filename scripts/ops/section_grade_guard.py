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
    from core.brokers import BrokerRuntimeConfig
    from core.licensing_api.default_connector import DefaultLicensingAPIConnector
    from core.licensing_api.grade_snapshot import build_grade_snapshot
    from core.licensing_api.models import LicensingTenantContext
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from core.brokers import BrokerRuntimeConfig
    from core.licensing_api.default_connector import DefaultLicensingAPIConnector
    from core.licensing_api.grade_snapshot import build_grade_snapshot
    from core.licensing_api.models import LicensingTenantContext
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "section_grade_guard_latest.json"
LOCAL_TENANT = LicensingTenantContext(
    tenant_id="local-ops",
    company_name="Local Operator",
    connector_name="default",
)

SECTION_COMMANDS: dict[str, list[list[str]]] = {
    "architecture_and_modularity": [
        ["./scripts/ops/opsctl.sh", "platform-control-plane", "--json"],
        ["./scripts/ops/opsctl.sh", "dashboard-refresh"],
    ],
    "live_trading_readiness": [
        ["./scripts/ops/opsctl.sh", "live-canary-control", "--json"],
        ["./scripts/ops/opsctl.sh", "incident-closeout", "--json"],
        ["./scripts/ops/opsctl.sh", "health"],
    ],
    "data_ingestion_and_storage": [
        ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
        ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
        ["./scripts/ops/opsctl.sh", "cost-telemetry", "--json"],
    ],
    "training_and_model_quality": [
        ["./scripts/ops/opsctl.sh", "training-quality", "--json"],
        ["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"],
        ["./scripts/ops/opsctl.sh", "coverage-gap-closer", "--json"],
    ],
    "security_governance_and_auditability": [
        ["./scripts/ops/opsctl.sh", "security-evidence-autofix", "--json"],
        ["./scripts/ops/opsctl.sh", "platform-control-plane", "--json"],
    ],
    "ops_and_autonomy": [
        ["./scripts/ops/opsctl.sh", "autonomy-control", "--json"],
        ["./scripts/ops/opsctl.sh", "incident-closeout", "--json"],
        ["./scripts/ops/opsctl.sh", "chrome-headless-guard", "--json"],
    ],
    "observability_and_reporting": [
        ["./scripts/ops/opsctl.sh", "dashboard-refresh"],
        ["./scripts/ops/opsctl.sh", "incident-timeline", "--json"],
        ["./scripts/ops/opsctl.sh", "incident-report", "--json"],
    ],
    "testing_and_qa": [
        ["./scripts/ops/opsctl.sh", "command-validity", "--json"],
        ["./scripts/ops/opsctl.sh", "platform-control-plane", "--json"],
    ],
    "api_and_partner_readiness": [
        ["./scripts/ops/opsctl.sh", "platform-control-plane", "--json"],
        ["./scripts/ops/opsctl.sh", "dashboard-refresh"],
    ],
    "portability_and_apple_silicon_optimization": [
        ["./scripts/ops/opsctl.sh", "cost-telemetry", "--json"],
        ["./scripts/ops/opsctl.sh", "apple-profile", "status"],
    ],
    "research_and_simulation_depth": [
        ["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"],
        ["./scripts/ops/opsctl.sh", "platform-control-plane", "--json"],
        ["./scripts/ops/opsctl.sh", "training-quality", "--json"],
    ],
}


def _row_from_section(slug: str, row: dict[str, Any]) -> dict[str, Any]:
    floor_state = str(row.get("floor_state") or "below_floor")
    return {
        "section": slug,
        "state": floor_state,
        "letter_grade": str(row.get("letter_grade") or ""),
        "raw_letter_grade": str(row.get("raw_letter_grade") or ""),
        "score": float(row.get("score", 0.0) or 0.0),
        "raw_score": float(row.get("raw_score", 0.0) or 0.0),
        "target_floor_letter_grade": str(row.get("target_floor_letter_grade") or ""),
        "target_floor_score": float(row.get("target_floor_score", 0.0) or 0.0),
        "floor_contract_active": bool(row.get("floor_contract_active", False)),
        "floor_reason": str(row.get("floor_reason") or "").strip(),
        "signals": dict(row.get("signals") or {}),
        "recommended_commands": [list(cmd) for cmd in SECTION_COMMANDS.get(slug, [])],
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    connector = DefaultLicensingAPIConnector()
    snapshot = build_grade_snapshot(
        project_root=project_root,
        runtime_config=BrokerRuntimeConfig.from_env(),
        tenant=LOCAL_TENANT,
        endpoint_count=len(connector.exposed_endpoints),
    )
    section_rows = [_row_from_section(slug, row) for slug, row in (snapshot.get("section_grades") or {}).items()]
    below_floor = [row["section"] for row in section_rows if row["state"] == "below_floor"]
    protected_by_floor = [row["section"] for row in section_rows if row["state"] == "protected_by_floor"]
    at_floor = [row["section"] for row in section_rows if row["state"] == "at_floor"]
    overall_status = "blocked" if below_floor else ("degraded" if protected_by_floor else "ready")
    recommended_actions = ordered_unique(
        [
            "keep the section-grade floor bot active so A-/A sections stay protected even when raw live artifacts dip during bounded recovery"
            if section_rows
            else "",
            "run the targeted repair commands for the floor-protected sections so their raw grades catch back up to the protected floor"
            if protected_by_floor
            else "",
            "focus on the sections below floor first; they are the only grades not meeting the current A-/A contract"
            if below_floor
            else "",
        ]
        + [row["floor_reason"] for row in section_rows if row["state"] == "protected_by_floor" and row["floor_reason"]]
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not below_floor,
        "overall_status": overall_status,
        "overall_letter_grade": str(snapshot.get("overall_letter_grade") or ""),
        "raw_overall_letter_grade": str(snapshot.get("raw_overall_letter_grade") or ""),
        "overall_score": float(snapshot.get("overall_score", 0.0) or 0.0),
        "raw_overall_score": float(snapshot.get("raw_overall_score", 0.0) or 0.0),
        "section_count": len(section_rows),
        "below_floor_count": len(below_floor),
        "protected_by_floor_count": len(protected_by_floor),
        "at_floor_count": len(at_floor),
        "below_floor_sections": below_floor,
        "protected_sections": protected_by_floor,
        "at_floor_sections": at_floor,
        "sections": section_rows,
        "grade_snapshot": snapshot,
        "upgrade_track": {
            "family": "infrabots",
            "upgradeable": True,
            "current_generation": "section_grade_floor_guard_v1",
            "co_managed_with": [
                "grade_regression_guard",
                "grade_regression_autopilot",
                "runtime_artifact_refresh",
            ],
            "future_upgrade_paths": [
                "per-section retry budgets instead of a shared floor policy",
                "partner-facing floor alerts when tenant readiness drops under contract",
                "floor-aware canary promotion gating that pauses release windows automatically",
            ],
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard the system's section-grade floor so each major section stays at A or A-.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "section_grade_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"below_floor_count={payload.get('below_floor_count', 0)} "
            f"protected_by_floor_count={payload.get('protected_by_floor_count', 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
