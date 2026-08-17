#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.master_grandmaster_evidence import synthesize_master_grandmaster_evidence
    from scripts.ops.long_runtime_common import write_payload
else:
    from core.master_grandmaster_evidence import synthesize_master_grandmaster_evidence
    from .long_runtime_common import PROJECT_ROOT, write_payload


DEFAULT_PATHS = {
    "policy": "config/master_grandmaster_evidence_v2.json",
    "organization_policy": "config/bot_organization_v1.json",
    "bot_organization_health": "governance/health/bot_organization_latest.json",
    "bot_hierarchy": "governance/bot_organization/bot_hierarchy_latest.json",
    "regime_context": "governance/health/regime_control_plane_latest.json",
    "paper_truth": "governance/health/paper_execution_truth_layer_latest.json",
    "profitability_evidence": "governance/health/profitability_evidence_firewall_latest.json",
    "source_verification": "governance/health/source_verification_latest.json",
    "runtime_throttle": "governance/health/runtime_throttle_control_latest.json",
    "account_positions": "governance/health/account_position_study_latest.json",
    "execution_calibration": "governance/health/paper_execution_calibration_latest.json",
    "sleeve_profitability": "governance/health/sleeve_profitability_dashboard_latest.json",
    "out_file": "governance/health/master_grandmaster_evidence_v2_latest.json",
    "packet_out": "governance/master_grandmaster/evidence_packets_v2_latest.json",
}


def _resolve(project_root: Path, raw: Path | None, key: str) -> Path:
    path = raw or Path(DEFAULT_PATHS[key])
    return path if path.is_absolute() else project_root / path


def _load_json_with_receipt(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError:
        return {}, ""
    receipt = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, receipt
    return (payload if isinstance(payload, dict) else {}), receipt


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    policy_path: Path | None = None,
    organization_policy_path: Path | None = None,
    bot_organization_health_path: Path | None = None,
    bot_hierarchy_path: Path | None = None,
    regime_context_path: Path | None = None,
    paper_truth_path: Path | None = None,
    profitability_evidence_path: Path | None = None,
    source_verification_path: Path | None = None,
    runtime_throttle_path: Path | None = None,
    account_positions_path: Path | None = None,
    execution_calibration_path: Path | None = None,
    sleeve_profitability_path: Path | None = None,
    packet_out_path: Path | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = project_root.resolve()
    paths = {
        "policy": _resolve(project_root, policy_path, "policy"),
        "organization_policy": _resolve(
            project_root, organization_policy_path, "organization_policy"
        ),
        "bot_organization_health": _resolve(
            project_root, bot_organization_health_path, "bot_organization_health"
        ),
        "bot_hierarchy": _resolve(project_root, bot_hierarchy_path, "bot_hierarchy"),
        "regime_context": _resolve(project_root, regime_context_path, "regime_context"),
        "paper_truth": _resolve(project_root, paper_truth_path, "paper_truth"),
        "profitability_evidence": _resolve(
            project_root, profitability_evidence_path, "profitability_evidence"
        ),
        "source_verification": _resolve(
            project_root, source_verification_path, "source_verification"
        ),
        "runtime_throttle": _resolve(
            project_root, runtime_throttle_path, "runtime_throttle"
        ),
        "account_positions": _resolve(
            project_root, account_positions_path, "account_positions"
        ),
        "execution_calibration": _resolve(
            project_root, execution_calibration_path, "execution_calibration"
        ),
        "sleeve_profitability": _resolve(
            project_root, sleeve_profitability_path, "sleeve_profitability"
        ),
    }
    loaded = {name: _load_json_with_receipt(path) for name, path in paths.items()}
    payloads = {name: row[0] for name, row in loaded.items()}
    file_receipts = {name: row[1] for name, row in sorted(loaded.items())}
    organization_policy = payloads["organization_policy"]
    result = synthesize_master_grandmaster_evidence(
        policy=payloads["policy"],
        regime_model=(organization_policy.get("regime_model") or {}),
        bot_organization_health=payloads["bot_organization_health"],
        bot_hierarchy=payloads["bot_hierarchy"],
        regime_payload=payloads["regime_context"],
        paper_truth=payloads["paper_truth"],
        profitability_evidence=payloads["profitability_evidence"],
        source_verification=payloads["source_verification"],
        runtime_throttle=payloads["runtime_throttle"],
        account_positions=payloads["account_positions"],
        execution_calibration=payloads["execution_calibration"],
        sleeve_profitability=payloads["sleeve_profitability"],
        now=now,
    )
    sleeve_masters = list(result.pop("sleeve_masters", []))
    packet_path = packet_out_path or _resolve(project_root, None, "packet_out")
    source_files = {name: str(path) for name, path in sorted(paths.items())}
    packet_receipt = hashlib.sha256(
        json.dumps(
            sleeve_masters,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    catalog = {
        "timestamp_utc": result.get("timestamp_utc"),
        "schema_version": 1,
        "policy_id": result.get("policy_id"),
        "operating_mode": result.get("operating_mode"),
        "overall_status": result.get("overall_status"),
        "structural_grade": result.get("structural_grade"),
        "sleeve_master_count": len(sleeve_masters),
        "organized_bot_count": result.get("organized_bot_count"),
        "packet_receipt_sha256": packet_receipt,
        "source_file_receipts": file_receipts,
        "sleeve_masters": sleeve_masters,
        "authority": result.get("authority") or {},
    }
    publication_receipt_input = {
        "core_evidence_receipt_sha256": str(
            (result.get("evidence_epoch") or {}).get("receipt_sha256") or ""
        ),
        "packet_receipt_sha256": packet_receipt,
        "source_file_receipts": file_receipts,
    }
    publication_receipt = hashlib.sha256(
        json.dumps(
            publication_receipt_input,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    catalog["publication_receipt_sha256"] = publication_receipt
    grand_master = result.get("grand_master") or {}
    result["sleeve_master_summary"] = {
        "catalog_path": str(packet_path),
        "packet_receipt_sha256": packet_receipt,
        "count": len(sleeve_masters),
        "status_counts": grand_master.get("sleeve_master_status_counts") or {},
        "grade_counts": grand_master.get("sleeve_master_grade_counts") or {},
    }
    result["source_files"] = source_files
    result["source_file_receipts"] = file_receipts
    result["publication_receipt"] = {
        "receipt_sha256": publication_receipt,
        **publication_receipt_input,
    }
    result["authority_contract"] = {
        "advisory_shadow_only": True,
        "creates_order_payloads": False,
        "mutates_runtime_or_registry": False,
        "automatic_live_promotion": False,
        "human_authorization_required_for_live": True,
    }
    return result, catalog


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize bounded sleeve-master and grand-master evidence without order authority."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--organization-policy", type=Path)
    parser.add_argument("--bot-organization-health", type=Path)
    parser.add_argument("--bot-hierarchy", type=Path)
    parser.add_argument("--regime-context", type=Path)
    parser.add_argument("--paper-truth", type=Path)
    parser.add_argument("--profitability-evidence", type=Path)
    parser.add_argument("--source-verification", type=Path)
    parser.add_argument("--runtime-throttle", type=Path)
    parser.add_argument("--account-positions", type=Path)
    parser.add_argument("--execution-calibration", type=Path)
    parser.add_argument("--sleeve-profitability", type=Path)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--packet-out", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    out_path = _resolve(project_root, args.out_file, "out_file")
    packet_out_path = _resolve(project_root, args.packet_out, "packet_out")
    health, catalog = build_payload(
        project_root,
        policy_path=args.policy,
        organization_policy_path=args.organization_policy,
        bot_organization_health_path=args.bot_organization_health,
        bot_hierarchy_path=args.bot_hierarchy,
        regime_context_path=args.regime_context,
        paper_truth_path=args.paper_truth,
        profitability_evidence_path=args.profitability_evidence,
        source_verification_path=args.source_verification,
        runtime_throttle_path=args.runtime_throttle,
        account_positions_path=args.account_positions,
        execution_calibration_path=args.execution_calibration,
        sleeve_profitability_path=args.sleeve_profitability,
        packet_out_path=packet_out_path,
    )
    write_payload(packet_out_path, catalog)
    write_payload(out_path, health)
    if args.json:
        print(json.dumps(health, ensure_ascii=True))
    else:
        print(
            "master_grandmaster_evidence_v2 "
            f"status={health['overall_status']} "
            f"structural_grade={health['structural_grade']} "
            f"evidence_grade={health['grade']} "
            f"paper_ready={int(bool(health['paper_coordination_ready']))} "
            f"human_live_review_ready={int(bool(health['human_live_review_evidence_ready']))} "
            f"masters={health['sleeve_master_count']}"
        )
    return 0 if health.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
