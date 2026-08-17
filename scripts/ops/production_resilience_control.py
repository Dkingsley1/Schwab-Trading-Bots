#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "production_resilience_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "production_resilience_control_latest.json"
EXPECTED_SECTION_IDS = tuple(f"{index:02d}_" for index in range(1, 11))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _grade(percent: float, *, complete: bool = False) -> str:
    if complete and percent >= 100.0:
        return "A+"
    if percent >= 90.0:
        return "A"
    if percent >= 80.0:
        return "B"
    if percent >= 70.0:
        return "C"
    if percent >= 60.0:
        return "D"
    return "F"


def _evaluate(section_id: str, payload: dict[str, Any]) -> tuple[bool, bool, bool, list[str], dict[str, Any]]:
    blockers: list[str] = []
    details: dict[str, Any] = {}
    implementation_ready = False
    evidence_ready = False
    paper_evidence_ready = True

    if section_id == "01_two_tier_self_healing":
        safety = _as_dict(payload.get("safety_contract"))
        implementation_ready = bool(
            safety.get("always_on_observation", False)
            and safety.get("heavy_maintenance_separated", False)
            and safety.get("exact_refresh_allowlist_only", False)
            and not safety.get("automatic_live_orders", True)
        )
        evidence_ready = bool(payload.get("ok", False) and not payload.get("blockers"))
        paper_evidence_ready = evidence_ready
        details = {"safety_contract": safety, "heavy_controller": payload.get("heavy_controller")}
    elif section_id == "02_honest_grade_semantics":
        grading = _as_dict(payload.get("grading_contract"))
        implementation_ready = bool(
            payload.get("semantic_ok_scope") == "control_implementation"
            and payload.get("raw_profitability_grade_overridden") is False
            and grading.get("generic_control_ok_must_not_be_interpreted_as_economic_readiness", False)
            and grading.get("future_profitability_is_not_guaranteed", False)
        )
        evidence_ready = implementation_ready
        paper_evidence_ready = implementation_ready
        details = {
            "raw_profitability_grade": payload.get("raw_profitability_grade"),
            "control_grade": payload.get("control_grade"),
            "economic_evidence_grade": payload.get("economic_evidence_grade"),
        }
    elif section_id == "03_exclusive_control_ownership":
        contract = _as_dict(payload.get("control_contract"))
        implementation_ready = bool(
            contract.get("one_declared_writer_per_resource", False)
            and contract.get("owners_are_source_backed", False)
            and contract.get("mutable_automation_is_coordinated", False)
        )
        evidence_ready = bool(payload.get("ok", False) and not payload.get("duplicate_resource_paths"))
        paper_evidence_ready = evidence_ready
        details = {"control_count": payload.get("control_count"), "duplicates": payload.get("duplicate_resource_paths")}
    elif section_id == "04_immutable_release_boundary":
        boundary = _as_dict(payload.get("immutable_release_boundary"))
        git_integrity = _as_dict(payload.get("git_integrity"))
        implementation_ready = bool(
            boundary.get("requires_clean_worktree", False)
            and boundary.get("requires_upstream_synchronization", False)
            and boundary.get("rollback_ready", False)
        )
        evidence_ready = bool(boundary.get("ready", False) and git_integrity.get("ready", False))
        details = {"boundary": boundary, "git_integrity": git_integrity}
    elif section_id == "05_scheduled_fault_injection":
        program = _as_dict(payload.get("drill_program"))
        required_count = int(payload.get("required_drill_count", 0) or 0)
        implementation_ready = bool(
            required_count == 10
            and program.get("automation_ready", False)
            and _as_dict(payload.get("evidence_scope")).get("live_execution_authority") is False
        )
        evidence_ready = bool(
            payload.get("ok", False)
            and int(payload.get("verified_drill_count", 0) or 0) == 10
            and not payload.get("overdue_drills")
            and not payload.get("recovery_slo_breaches")
            and program.get("recovery_slo_met", False)
        )
        details = {"verified": payload.get("verified_drill_count"), "required": required_count, "program": program}
    elif section_id == "06_bounded_repair_circuits":
        bounded = _as_dict(payload.get("bounded_repair"))
        implementation_ready = bool(
            int(bounded.get("max_actions_per_cycle", 0) or 0) > 0
            and float(bounded.get("cooldown_seconds", 0.0) or 0.0) > 0.0
            and int(bounded.get("max_failures_before_circuit", 0) or 0) > 0
            and float(bounded.get("circuit_open_seconds", 0.0) or 0.0) > 0.0
        )
        evidence_ready = bool(implementation_ready and not bounded.get("open_circuits") and payload.get("ok", False))
        paper_evidence_ready = evidence_ready
        details = bounded
    elif section_id == "07_transactional_trading_truth":
        contract = _as_dict(payload.get("contract"))
        integrity = _as_dict(payload.get("integrity"))
        implementation_ready = bool(
            contract.get("transactional_reservation_before_submit", False)
            and contract.get("ambiguous_submit_never_auto_retried", False)
            and contract.get("foreign_key_integrity_required", False)
            and contract.get("wal_full_sync_required", False)
            and contract.get("event_state_materialization_must_match", False)
        )
        evidence_ready = bool(payload.get("ok", False) and integrity.get("ok", False))
        paper_evidence_ready = evidence_ready
        details = {"integrity": integrity, "unresolved_intent_count": payload.get("unresolved_intent_count")}
    elif section_id == "08_measured_data_recoverability":
        objectives = _as_dict(payload.get("recovery_objectives"))
        implementation_ready = bool(
            _as_dict(objectives.get("rpo")).get("target_minutes")
            and _as_dict(objectives.get("rto")).get("target_seconds")
            and objectives.get("evidence_receipt_sha256")
            and objectives.get("paper_collection_blocked") is False
        )
        evidence_ready = bool(objectives.get("ready", False))
        details = objectives
    elif section_id == "09_independent_deadman_monitor":
        boundary = _as_dict(payload.get("implementation_boundary"))
        implementation_ready = bool(
            boundary.get("stdlib_only", False)
            and boundary.get("imports_trading_runtime") is False
            and boundary.get("runs_as_separate_launchd_process", False)
            and boundary.get("automatic_repairs") is False
            and boundary.get("automatic_orders") is False
        )
        evidence_ready = bool(payload.get("production_monitor_ready", False))
        paper_evidence_ready = bool(payload.get("local_monitor_ready", False))
        details = {"implementation_boundary": boundary, "off_host_delivery": payload.get("off_host_delivery")}
    elif section_id == "10_honest_profitability_evidence":
        grading = _as_dict(payload.get("grading_contract"))
        implementation_ready = bool(
            payload.get("control_implementation_ready", False)
            and str(payload.get("control_grade") or "") == "A+"
            and payload.get("raw_profitability_grade_overridden") is False
            and grading.get("future_profitability_is_not_guaranteed", False)
        )
        evidence_ready = bool(payload.get("live_promotion_ready", False) and payload.get("promotion_evidence_ready", False))
        details = {
            "control_grade": payload.get("control_grade"),
            "economic_evidence_grade": payload.get("economic_evidence_grade"),
            "raw_profitability_grade": payload.get("raw_profitability_grade"),
            "economic_evidence_blockers": payload.get("economic_evidence_blockers"),
        }
    else:
        blockers.append("unknown_resilience_section")

    if not implementation_ready:
        blockers.append("implementation_contract_not_ready")
    if not evidence_ready:
        blockers.append("production_evidence_not_ready")
    return implementation_ready, evidence_ready, paper_evidence_ready, ordered_unique(blockers), details


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name
    config = load_json(config_path)
    current = now or datetime.now(timezone.utc)
    specs = [row for row in config.get("sections", []) if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    framework_blockers: list[str] = []
    prefixes = tuple(str(row.get("section_id") or "")[:3] for row in specs)
    if len(specs) != 10 or prefixes != EXPECTED_SECTION_IDS:
        framework_blockers.append("ten_section_framework_manifest_invalid")

    for spec in specs:
        section_id = str(spec.get("section_id") or "").strip()
        artifact_path = project_root / str(spec.get("artifact") or "")
        owner_path = project_root / str(spec.get("owner_source") or "")
        payload = load_json(artifact_path)
        exists = bool(artifact_path.is_file() and payload)
        age = payload_age_minutes(payload, artifact_path, now=current) if exists else None
        fresh = bool(age is not None and age <= float(spec.get("max_age_minutes", 0.0) or 0.0))
        implementation_ready, evidence_ready, paper_evidence_ready, blockers, details = _evaluate(section_id, payload)
        if not owner_path.is_file():
            implementation_ready = False
            blockers.append("owner_source_missing")
        if not exists:
            evidence_ready = False
            paper_evidence_ready = False
            blockers.append("artifact_missing")
        elif not fresh:
            evidence_ready = False
            paper_evidence_ready = False
            blockers.append("artifact_stale")
        paper_required = bool(spec.get("paper_required", False))
        paper_ready = bool(implementation_ready and (paper_evidence_ready if paper_required else True))
        live_ready = bool(implementation_ready and evidence_ready and fresh)
        rows.append(
            {
                "section_id": section_id,
                "title": str(spec.get("title") or section_id),
                "paper_required": paper_required,
                "artifact": str(artifact_path),
                "owner_source": str(spec.get("owner_source") or ""),
                "age_minutes": round(float(age), 4) if age is not None else None,
                "max_age_minutes": float(spec.get("max_age_minutes", 0.0) or 0.0),
                "fresh": fresh,
                "implementation_ready": implementation_ready,
                "evidence_ready": evidence_ready,
                "paper_ready": paper_ready,
                "live_ready": live_ready,
                "blockers": ordered_unique(blockers),
                "details": details,
                "source_sha256": _sha256(artifact_path) if exists else "",
            }
        )

    implementation_count = sum(1 for row in rows if row["implementation_ready"])
    live_count = sum(1 for row in rows if row["live_ready"])
    paper_required_rows = [row for row in rows if row["paper_required"]]
    paper_ready_count = sum(1 for row in paper_required_rows if row["paper_ready"])
    implementation_ready = bool(len(rows) == 10 and implementation_count == 10 and not framework_blockers)
    paper_ready = bool(implementation_ready and paper_ready_count == len(paper_required_rows))
    live_ready = bool(implementation_ready and live_count == 10)
    implementation_percent = 10.0 * implementation_count
    paper_percent = 100.0 * paper_ready_count / max(len(paper_required_rows), 1)
    live_percent = 10.0 * live_count
    overall_status = "ready" if live_ready else "paper_ready_with_live_evidence_debt" if paper_ready else "blocked"
    receipt_input = {
        "config_sha256": _sha256(config_path),
        "sources": {row["section_id"]: row["source_sha256"] for row in rows},
    }
    receipt = hashlib.sha256(
        json.dumps(receipt_input, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    paper_blockers = ordered_unique(
        framework_blockers
        + [f"{row['section_id']}:{item}" for row in rows if not row["paper_ready"] for item in row["blockers"]]
    )
    live_blockers = ordered_unique(
        framework_blockers
        + [f"{row['section_id']}:{item}" for row in rows if not row["live_ready"] for item in row["blockers"]]
    )
    return {
        "timestamp_utc": current.isoformat(),
        "schema_version": 1,
        "policy_id": str(config.get("policy_id") or ""),
        "ok": paper_ready,
        "overall_status": overall_status,
        "framework_awareness_ready": implementation_ready,
        "implementation_ready": implementation_ready,
        "implementation_grade": _grade(implementation_percent, complete=implementation_ready),
        "implementation_percent": implementation_percent,
        "paper_soak_ready": paper_ready,
        "paper_soak_grade": _grade(paper_percent, complete=paper_ready),
        "paper_soak_readiness_percent": round(paper_percent, 2),
        "live_promotion_ready": live_ready,
        "live_promotion_grade": _grade(live_percent, complete=live_ready),
        "live_promotion_readiness_percent": live_percent,
        "section_count": len(rows),
        "implementation_ready_section_count": implementation_count,
        "paper_required_section_count": len(paper_required_rows),
        "paper_ready_section_count": paper_ready_count,
        "live_ready_section_count": live_count,
        "sections": rows,
        "paper_blockers": paper_blockers,
        "live_blockers": live_blockers,
        "evidence_epoch": {
            "id": f"production-resilience:{receipt[:16]}",
            "receipt_sha256": receipt,
            "config_sha256": receipt_input["config_sha256"],
        },
        "authority_contract": {
            "live_execution_authority": False,
            "automatic_unlock_allowed": False,
            "operator_release_still_required_after_all_sections_pass": True,
            "future_profitability_guaranteed": False,
        },
        "recommended_actions": ordered_unique(
            [
                "keep paper collection running while only production evidence remains" if paper_ready and not live_ready else "",
                "create a clean synchronized release and exact rollback manifest" if any(row["section_id"].startswith("04_") and not row["live_ready"] for row in rows) else "",
                "configure and verify an off-host independent-monitor receiver" if any(row["section_id"].startswith("09_") and not row["live_ready"] for row in rows) else "",
                "continue collecting independent post-cost profitability evidence without changing the raw grade" if any(row["section_id"].startswith("10_") and not row["live_ready"] for row in rows) else "",
                "refresh verified RPO and RTO evidence" if any(row["section_id"].startswith("08_") and not row["live_ready"] for row in rows) else "",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate the ten production-resilience hardening contracts.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config or Path("config/production_resilience_v1.json")
    out_path = args.out_file or Path("governance/health/production_resilience_control_latest.json")
    config_path = config_path if config_path.is_absolute() else project_root / config_path
    out_path = out_path if out_path.is_absolute() else project_root / out_path
    payload = build_payload(project_root, config_path=config_path)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "production_resilience_control "
            f"status={payload['overall_status']} paper={payload['paper_soak_readiness_percent']} "
            f"live={payload['live_promotion_readiness_percent']}"
        )
    return 0 if payload.get("paper_soak_ready", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
