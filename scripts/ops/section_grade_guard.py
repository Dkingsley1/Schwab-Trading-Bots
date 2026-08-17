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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from core.brokers import BrokerRuntimeConfig
    from core.licensing_api.default_connector import DefaultLicensingAPIConnector
    from core.licensing_api.grade_snapshot import build_grade_snapshot
    from core.licensing_api.models import LicensingTenantContext
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


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
        [
            "./scripts/ops/opsctl.sh",
            "storage-backpressure-autopilot",
            "--apply",
            "--quick-bounded",
            "--json",
        ],
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
PAPER_SOAK_ADVISORY_BELOW_FLOOR_SECTIONS = {
    "live_trading_readiness",
    "training_and_model_quality",
    "ops_and_autonomy",
}
STORAGE_PAPER_SOAK_ADVISORY_SECTION = "data_ingestion_and_storage"
STORAGE_SOAK_RAW_LIVE_MAX_CORE_LINES = 10_000
STORAGE_SOAK_RAW_LIVE_MAX_TOTAL_LINES = 15_000
STORAGE_SOAK_RAW_LIVE_MAX_AGE_SECONDS = 900.0
STORAGE_SOAK_BOUNDED_PRESSURE_MAX = 1.0
GUARDED_READ_ONLY_RUNTIME_STATES = {
    "guarded_live_read_only",
    "managed_cold_lane_deferred",
    "managed_coverage_stage_deferred",
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


def _guarded_paper_ready(health_fast: dict[str, Any]) -> bool:
    operational = health_fast.get("operational_readiness") if isinstance(health_fast.get("operational_readiness"), dict) else {}
    guarded_paper = operational.get("guarded_paper") if isinstance(operational.get("guarded_paper"), dict) else {}
    return bool(
        guarded_paper.get("ok", False)
        and str(guarded_paper.get("status") or "").strip().lower() in {"ready", "armed", "guarded_ready"}
    )


def _live_execution_locked(health_fast: dict[str, Any], runtime: dict[str, Any]) -> bool:
    operational = health_fast.get("operational_readiness") if isinstance(health_fast.get("operational_readiness"), dict) else {}
    live_execution = operational.get("live_execution") if isinstance(operational.get("live_execution"), dict) else {}
    clearance_state = str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower()
    return bool(
        clearance_state in GUARDED_READ_ONLY_RUNTIME_STATES
        or str(live_execution.get("status") or "").strip().lower() in {"blocked_read_only", "read_only", "operator_gated"}
        or "live_execution_requires_explicit_operator_control" in set(live_execution.get("blockers") or [])
    )


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _raw_live_storage_backlog_clear(storage: dict[str, Any]) -> bool:
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    effective = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), dict) else {}
    estimate = effective.get("raw_live_estimate") if isinstance(effective.get("raw_live_estimate"), dict) else {}
    raw_live = estimate or (backpressure.get("raw_live") if isinstance(backpressure.get("raw_live"), dict) else {})
    if not raw_live:
        return True
    return bool(
        _safe_int(raw_live.get("core_pending_lines"), 0) <= STORAGE_SOAK_RAW_LIVE_MAX_CORE_LINES
        and _safe_int(raw_live.get("total_pending_lines"), _safe_int(raw_live.get("core_pending_lines"), 0))
        <= STORAGE_SOAK_RAW_LIVE_MAX_TOTAL_LINES
        and _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0) <= STORAGE_SOAK_RAW_LIVE_MAX_AGE_SECONDS
    )


def _storage_section_advisory_for_paper_soak(project_root: Path) -> bool:
    storage = load_json(project_root / "governance" / "health" / "ingestion_storage_control_latest.json")
    status = str(storage.get("overall_status") or storage.get("status") or "").strip().lower()
    severity = str(storage.get("severity") or "").strip().lower()
    if status not in {"ready", "ok", "advisory"}:
        return False
    pressure_index = _safe_float(storage.get("pressure_index"), 1.0)
    soak_contract = (
        storage.get("continuous_run_soak_contract")
        if isinstance(storage.get("continuous_run_soak_contract"), dict)
        else {}
    )
    blockers = {
        str(item or "").strip()
        for item in (soak_contract.get("blockers") if isinstance(soak_contract.get("blockers"), list) else [])
        if str(item or "").strip()
    }
    soak_ready = bool(soak_contract.get("ready", False) or soak_contract.get("soak_ready", False))
    bounded = storage.get("bounded_recovery_contract") if isinstance(storage.get("bounded_recovery_contract"), dict) else {}
    integrity = storage.get("data_integrity") if isinstance(storage.get("data_integrity"), dict) else {}
    writer = storage.get("writer_shedding") if isinstance(storage.get("writer_shedding"), dict) else {}
    efficiency = storage.get("storage_efficiency_contract") if isinstance(storage.get("storage_efficiency_contract"), dict) else {}
    raw_live_clear = _raw_live_storage_backlog_clear(storage)
    bounded_steady_state_ready = bool(
        severity in {"", "ready", "stable", "low", "normal", "watch", "elevated"}
        and pressure_index <= 0.85
        and bool(bounded.get("route_verified", False))
        and not bool(bounded.get("hard_gate_active", False))
        and not bool(bounded.get("effective_hard_gate_active", False))
        and not writer.get("hard_breaches")
        and not writer.get("elevated_breaches")
        and all(
            _safe_int(integrity.get(key), 0) == 0
            for key in (
                "sql_invalid_lines",
                "sql_overlay_invalid_lines",
                "sql_overlay_oversize_payloads",
                "sql_overlay_ops_write_failures",
            )
        )
        and str(efficiency.get("overall_status") or "ready").strip().lower() in {"ready", "ok"}
        and str(efficiency.get("grade") or "A").strip().upper() in {"A", "A+"}
        and raw_live_clear
    )
    bounded_transient_ready = bool(
        pressure_index <= STORAGE_SOAK_BOUNDED_PRESSURE_MAX
        and bool(bounded.get("route_verified", False))
        and bool(bounded.get("active_drain_progress", False) or bounded.get("drain_delta_signal_observed", False))
        and not bool(bounded.get("hard_gate_active", False))
        and not bool(bounded.get("effective_hard_gate_active", False))
        and not writer.get("hard_breaches")
        and not writer.get("elevated_breaches")
        and all(
            _safe_int(integrity.get(key), 0) == 0
            for key in (
                "sql_invalid_lines",
                "sql_overlay_invalid_lines",
                "sql_overlay_oversize_payloads",
                "sql_overlay_ops_write_failures",
            )
        )
        and str(efficiency.get("overall_status") or "ready").strip().lower() in {"ready", "ok"}
        and str(efficiency.get("grade") or "A").strip().upper() in {"A", "A+"}
        and blockers.issubset({"steady_state_targets_not_clear"})
    )
    if severity not in {"", "ready", "stable", "low", "normal"} and not bounded_steady_state_ready and not bounded_transient_ready:
        return False
    if pressure_index > 0.50 and not bounded_transient_ready and not bounded_steady_state_ready:
        return False
    if (blockers or not soak_ready) and not bounded_transient_ready and not bounded_steady_state_ready:
        return False
    return raw_live_clear


def _advisory_below_floor_sections(
    below_floor: list[str],
    *,
    project_root: Path,
    guarded_paper_ready: bool,
    live_execution_locked: bool,
    guarded_paper_strict_clear: bool,
    overall_score: float,
) -> list[str]:
    if not (below_floor and guarded_paper_ready and live_execution_locked):
        return []
    advisory: list[str] = []
    default_advisory_allowed = bool(guarded_paper_strict_clear or overall_score >= 96.0)
    storage_advisory_allowed = _storage_section_advisory_for_paper_soak(project_root)
    for section in below_floor:
        if section in PAPER_SOAK_ADVISORY_BELOW_FLOOR_SECTIONS and default_advisory_allowed:
            advisory.append(section)
        elif section == STORAGE_PAPER_SOAK_ADVISORY_SECTION and storage_advisory_allowed:
            advisory.append(section)
    return advisory


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    connector = DefaultLicensingAPIConnector()
    snapshot = build_grade_snapshot(
        project_root=project_root,
        runtime_config=BrokerRuntimeConfig.from_env(),
        tenant=LOCAL_TENANT,
        endpoint_count=len(connector.exposed_endpoints),
    )
    health_fast = load_json(health_root / "health_fast_latest.json")
    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    section_rows = [_row_from_section(slug, row) for slug, row in (snapshot.get("section_grades") or {}).items()]
    below_floor = [row["section"] for row in section_rows if row["state"] == "below_floor"]
    protected_by_floor = [row["section"] for row in section_rows if row["state"] == "protected_by_floor"]
    at_floor = [row["section"] for row in section_rows if row["state"] == "at_floor"]
    guarded_paper_ready = _guarded_paper_ready(health_fast)
    live_execution_locked = _live_execution_locked(health_fast, runtime)
    guarded_paper_strict_clear = bool(health_fast.get("strict_all_clear", False) and guarded_paper_ready and live_execution_locked)
    advisory_below_floor = _advisory_below_floor_sections(
        below_floor,
        project_root=project_root,
        guarded_paper_ready=guarded_paper_ready,
        live_execution_locked=live_execution_locked,
        guarded_paper_strict_clear=guarded_paper_strict_clear,
        overall_score=_safe_float(snapshot.get("overall_score"), 0.0),
    )
    paper_soak_advisory_below_floor = bool(below_floor and set(below_floor).issubset(set(advisory_below_floor)))
    blocking_below_floor = [section for section in below_floor if section not in set(advisory_below_floor)]
    overall_status = "blocked" if blocking_below_floor else ("degraded" if protected_by_floor else "ready")
    recommended_actions = ordered_unique(
        [
            "keep the section-grade floor bot active so A-/A sections stay protected even when raw live artifacts dip during bounded recovery"
            if section_rows
            else "",
            "run the targeted repair commands for the floor-protected sections so their raw grades catch back up to the protected floor"
            if protected_by_floor
            else "",
            "keep training collection, rebalancing, and coverage staging active; training quality debt is advisory while guarded paper is ready and live execution remains locked"
            if advisory_below_floor
            else "",
            "focus on the sections below floor first; they are the only grades not meeting the current A-/A contract"
            if blocking_below_floor
            else "",
        ]
        + [row["floor_reason"] for row in section_rows if row["state"] == "protected_by_floor" and row["floor_reason"]]
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not blocking_below_floor,
        "overall_status": overall_status,
        "overall_letter_grade": str(snapshot.get("overall_letter_grade") or ""),
        "raw_overall_letter_grade": str(snapshot.get("raw_overall_letter_grade") or ""),
        "overall_score": float(snapshot.get("overall_score", 0.0) or 0.0),
        "raw_overall_score": float(snapshot.get("raw_overall_score", 0.0) or 0.0),
        "section_count": len(section_rows),
        "below_floor_count": len(below_floor),
        "blocking_below_floor_count": len(blocking_below_floor),
        "advisory_below_floor_count": len(advisory_below_floor),
        "protected_by_floor_count": len(protected_by_floor),
        "at_floor_count": len(at_floor),
        "below_floor_sections": below_floor,
        "blocking_below_floor_sections": blocking_below_floor,
        "advisory_below_floor_sections": advisory_below_floor,
        "protected_sections": protected_by_floor,
        "at_floor_sections": at_floor,
        "sections": section_rows,
        "grade_snapshot": snapshot,
        "guarded_paper_ready": guarded_paper_ready,
        "live_execution_locked": live_execution_locked,
        "guarded_paper_strict_clear": guarded_paper_strict_clear,
        "paper_soak_advisory_below_floor": paper_soak_advisory_below_floor,
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
