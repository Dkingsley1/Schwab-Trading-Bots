#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json"


def _gate_ok(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    if "ok" in payload:
        return bool(payload.get("ok"))
    status = str(payload.get("overall_status") or "").strip().lower()
    if not status:
        return False
    return status not in {"blocked", "critical", "failed"}


def _gate_repair_rows(project_root: Path, *, gate_failures: list[str], coverage_shortfall_bots: int, quality_ok: bool, pipeline_ok: bool) -> list[dict[str, Any]]:
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    gate_specs: dict[str, dict[str, Any]] = {
        "training_success_confirmed": {
            "source_path": str(health_root / "training_success_latest.json"),
            "repair_hint": "rerun the failed retrain lane and refresh training success before promotion is reconsidered",
            "recommended_command": ["./scripts/ops/opsctl.sh", "retrain-force-targeted", "--json"],
        },
        "feature_store_manifest_strict_ok": {
            "source_path": str(project_root / "governance" / "feature_store" / "latest.json"),
            "repair_hint": "refresh the feature-store manifest and restore strict point-in-time contract readiness",
            "recommended_command": ["./scripts/ops/opsctl.sh", "feature-store", "--json"],
        },
        "retrain_schema_compatibility_ok": {
            "source_path": str(health_root / "retrain_schema_compatibility_latest.json"),
            "repair_hint": "refresh the retrain schema compatibility baseline and migration manifest before promoting schema-sensitive models",
            "recommended_command": ["./scripts/ops/opsctl.sh", "schema-migration", "--json"],
        },
    }
    rows: list[dict[str, Any]] = []
    for gate in gate_failures:
        spec = gate_specs.get(gate)
        if not spec:
            continue
        rows.append(
            {
                "gate": gate,
                "severity": "critical",
                "source_path": str(spec.get("source_path") or ""),
                "repair_hint": str(spec.get("repair_hint") or ""),
                "recommended_command": list(spec.get("recommended_command") or []),
            }
        )
    if coverage_shortfall_bots > 0:
        rows.append(
            {
                "gate": "walk_forward_coverage",
                "severity": "critical",
                "source_path": str(walk_root / "promotion_readiness_latest.json"),
                "repair_hint": f"clear the remaining walk-forward shortfall for {coverage_shortfall_bots} bots before any challenger can advance",
                "recommended_command": ["./scripts/ops/opsctl.sh", "coverage-gap-closer", "--apply-stage", "--auto-launch-off-hours", "--json"],
            }
        )
    if not quality_ok:
        rows.append(
            {
                "gate": "promotion_quality_gate",
                "severity": "warning",
                "source_path": str(health_root / "promotion_quality_gate_latest.json"),
                "repair_hint": "refresh the promotion quality gate after the training, replay, and schema blockers are repaired",
                "recommended_command": [str(project_root / "scripts" / "promotion_quality_gate.py"), "--json"],
            }
        )
    if not pipeline_ok:
        rows.append(
            {
                "gate": "promotion_pipeline",
                "severity": "warning",
                "source_path": str(walk_root / "promotion_pipeline_latest.json"),
                "repair_hint": "rerun the promotion pipeline after readiness and quality gates stop blocking",
                "recommended_command": [str(project_root / "scripts" / "ops" / "promotion_pipeline.py"), "--json"],
            }
        )
    return rows


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    champion_root = project_root / "governance" / "champion_challenger"
    walk_root = project_root / "governance" / "walk_forward"
    health_root = project_root / "governance" / "health"

    packet_path = champion_root / "promotion_packet_latest.json"
    readiness_path = walk_root / "promotion_readiness_latest.json"
    quality_path = health_root / "promotion_quality_gate_latest.json"
    pipeline_path = walk_root / "promotion_pipeline_latest.json"
    coverage_seed_path = walk_root / "coverage_seed_latest.json"
    coverage_gap_closer_path = walk_root / "coverage_gap_closer_latest.json"

    packet = load_json(packet_path)
    readiness = load_json(readiness_path)
    quality = load_json(quality_path)
    pipeline = load_json(pipeline_path)
    coverage_seed = load_json(coverage_seed_path)
    coverage_gap_closer = load_json(coverage_gap_closer_path)

    packet_complete = bool(
        packet.get("packet_complete", False)
        or packet.get("ready_for_committee", False)
        or packet.get("ok", False)
    )
    packet_sha256 = str(packet.get("packet_sha256") or "").strip()
    signature = packet.get("signature") if isinstance(packet.get("signature"), dict) else {}
    signature_verified = bool(signature.get("verified", False))
    gate_results = packet.get("gate_results") if isinstance(packet.get("gate_results"), dict) else {}
    gate_failures = sorted(key for key, value in gate_results.items() if not bool(value))
    promote_ok = bool(readiness.get("promote_ok", False))
    quality_ok = _gate_ok(quality)
    pipeline_ok = _gate_ok(pipeline)
    raw_coverage_shortfall_bots = int(readiness.get("coverage_shortfall_bots", 0) or 0)
    coverage_seed_queue = coverage_seed.get("seed_queue") if isinstance(coverage_seed.get("seed_queue"), list) else []
    coverage_seed_queue_size = int(((coverage_seed.get("standing_queue") or {}).get("seed_queue_size", 0) or 0))
    gap_autopilot = (
        coverage_gap_closer.get("autopilot_contract")
        if isinstance(coverage_gap_closer.get("autopilot_contract"), dict)
        else {}
    )
    gap_launch_state = str(gap_autopilot.get("launch_state") or "").strip().lower()
    stage_candidate_count = int(
        coverage_gap_closer.get(
            "staged_candidate_count",
            len(coverage_gap_closer.get("active_stage_candidates") or []),
        )
        or 0
    )
    queued_coverage_candidates = max(coverage_seed_queue_size, len(coverage_seed_queue), stage_candidate_count)
    coverage_seed_ready = bool(
        raw_coverage_shortfall_bots > 0
        and queued_coverage_candidates >= raw_coverage_shortfall_bots
        and str(gap_autopilot.get("overall_status") or "").strip().lower() in {"ready", "degraded"}
        and gap_launch_state in {
            "ready_to_launch",
            "auto_launch_off_hours_ready",
            "armed_for_off_hours_auto_launch",
            "stage_only_off_hours",
            "waiting_for_idle",
        }
        and (
            gap_launch_state != "waiting_for_idle"
            or bool(gap_autopilot.get("can_apply_stage", False))
        )
    )
    coverage_shortfall_bots = 0 if coverage_seed_ready else raw_coverage_shortfall_bots
    readiness_blocking_reasons = [
        str(item).strip().lower() for item in (readiness.get("blocking_reasons") or []) if str(item).strip()
    ]
    promote_ok_effective = bool(
        promote_ok
        or (
            coverage_seed_ready
            and readiness_blocking_reasons in (["insufficient_walk_forward_coverage"], [])
        )
    )
    signing_key_path = champion_root / "promotion_packet_signing_key.txt"
    signature_status = str(signature.get("status") or "")
    env_signing_key_present = bool(str(os.getenv("PROMOTION_PACKET_SIGNING_KEY", "") or "").strip())
    signing_key_present = signing_key_path.exists() and bool(signing_key_path.read_text(encoding="utf-8").strip()) if signing_key_path.exists() else False
    signing_material_ready = bool(signing_key_present or env_signing_key_present or signature_verified)
    source_count = len(packet.get("sources") or {}) if isinstance(packet.get("sources"), dict) else 0
    committee_packet_seed_ready = bool(
        packet.get("committee_packet_seed_ready", False)
        or (
            packet_sha256
            and source_count > 0
        )
    )
    readiness_repair_rows = _gate_repair_rows(
        project_root,
        gate_failures=gate_failures,
        coverage_shortfall_bots=coverage_shortfall_bots,
        quality_ok=quality_ok,
        pipeline_ok=pipeline_ok,
    )
    critical_repair_gate_count = sum(1 for row in readiness_repair_rows if str(row.get("severity") or "") == "critical")
    warning_repair_gate_count = sum(1 for row in readiness_repair_rows if str(row.get("severity") or "") == "warning")
    pipeline_steps = pipeline.get("steps") if isinstance(pipeline.get("steps"), list) else []
    coverage_only_pipeline_failures = bool(
        pipeline_steps
        and all(
            str((row.get("step") if isinstance(row, dict) else "") or "").strip()
            in {
                "walk_forward_promotion_gate",
                "lane_promotion_gate",
                "promotion_readiness_summary",
                "promotion_bottleneck_focus",
                "schema_migration_guard",
                "bot_support_owner_guard",
                "run_master_bot",
            }
            for row in pipeline_steps
            if isinstance(row, dict) and not bool(row.get("ok", False))
        )
    )
    pipeline_repairable_for_canary = bool(coverage_seed_ready and coverage_only_pipeline_failures)
    canary_packet_ready = bool(
        packet_complete
        and signature_verified
        and quality_ok
        and (promote_ok_effective or coverage_seed_ready)
        and (pipeline_ok or pipeline_repairable_for_canary)
    )

    autopilot_state = "blocked"
    if not packet:
        autopilot_state = "missing_packet"
    elif not packet_complete:
        autopilot_state = "repairing_readiness" if bool(signature_verified) else "assembling_packet"
    elif not signing_material_ready:
        autopilot_state = "awaiting_signing_material"
    elif not signature_verified:
        autopilot_state = "awaiting_signature"
    elif canary_packet_ready and not promote_ok:
        autopilot_state = "ready_for_supervised_canary"
    elif not promote_ok_effective or not quality_ok or not (pipeline_ok or pipeline_repairable_for_canary):
        autopilot_state = "repairing_readiness"
    else:
        autopilot_state = "awaiting_approval"

    blockers = ordered_unique(
        gate_failures
        + ([f"coverage_shortfall_bots={coverage_shortfall_bots}"] if coverage_shortfall_bots > 0 else [])
        + ([f"promotion_readiness:{','.join(str(item) for item in readiness.get('blocking_reasons') or [])}"] if not promote_ok_effective else [])
        + (["promotion_quality_gate_failed"] if not quality_ok else [])
        + (["promotion_pipeline_failed"] if not (pipeline_ok or pipeline_repairable_for_canary) else [])
        + (["promotion_packet_missing"] if not packet else [])
        + (["promotion_packet_incomplete"] if packet and not packet_complete else [])
        + (["promotion_packet_signature_unverified"] if packet and not signature_verified else [])
    )

    rollback_bundle = packet.get("rollback_bundle") if isinstance(packet.get("rollback_bundle"), dict) else {}
    rollback_reference = str(rollback_bundle.get("rollback_reference") or "")
    rollback_command = str(rollback_bundle.get("rollback_command") or "")
    signed_bundle_contract = {
        "packet_sha256": packet_sha256,
        "signature_verified": signature_verified,
        "signature_status": signature_status,
        "rollback_reference": rollback_reference,
        "rollback_command": rollback_command,
        "rollback_ready": bool(rollback_reference and rollback_command),
        "source_hash_count": len(packet.get("sources") or {}) if isinstance(packet.get("sources"), dict) else 0,
    }
    signability_contract = {
        "key_expected_path": str(signing_key_path),
        "key_present": signing_key_present,
        "env_key_present": env_signing_key_present,
        "signing_material_ready": signing_material_ready,
        "signature_status": signature_status,
        "packet_builder_command": [
            str(project_root / "scripts" / "promotion_packet_builder.py"),
            "--out-file",
            str(packet_path),
        ],
        "committee_packet_ready": bool(packet_complete and signature_verified),
        "committee_packet_seed_ready": committee_packet_seed_ready,
        "seeded_source_count": source_count,
        "critical_repair_gate_count": critical_repair_gate_count,
        "warning_repair_gate_count": warning_repair_gate_count,
        "can_sign_now": bool(packet_complete and signing_material_ready),
        "signing_material_source": (
            "packet_signature"
            if signature_verified
            else ("env:PROMOTION_PACKET_SIGNING_KEY" if env_signing_key_present else (str(signing_key_path) if signing_key_present else "missing"))
        ),
    }
    evidence_bundle = {
        "packet_path": str(packet_path),
        "readiness_path": str(readiness_path),
        "quality_gate_path": str(quality_path),
        "pipeline_path": str(pipeline_path),
        "source_count": len(packet.get("sources") or {}) if isinstance(packet.get("sources"), dict) else 0,
        "source_paths": sorted(str(value) for value in (packet.get("sources") or {}).values()) if isinstance(packet.get("sources"), dict) else [],
    }
    approval_record = {
        "approval_required": True,
        "approval_state": ("awaiting_operator_signoff" if autopilot_state == "awaiting_approval" else "not_ready"),
        "packet_sha256": packet_sha256,
        "signature_status": signature_status,
        "candidate_commit": str(((packet.get("code") or {}).get("git_commit") or "")),
        "rollback_reference": rollback_reference,
        "rollback_command": rollback_command,
        "approval_record_path": str(champion_root / "promotion_approval_record_latest.json"),
        "approval_record_seed_ready": bool(packet_complete and signing_material_ready),
        "committee_packet_seed_ready": committee_packet_seed_ready,
    }
    readiness_repair_contract = {
        "repairable_gate_count": len(readiness_repair_rows),
        "critical_repair_gate_count": critical_repair_gate_count,
        "warning_repair_gate_count": warning_repair_gate_count,
        "repair_rows": readiness_repair_rows,
        "coverage_shortfall_bots": coverage_shortfall_bots,
        "raw_coverage_shortfall_bots": raw_coverage_shortfall_bots,
        "training_success_confirmed": "training_success_confirmed" not in gate_failures,
        "feature_store_manifest_strict_ok": "feature_store_manifest_strict_ok" not in gate_failures,
        "retrain_schema_compatibility_ok": "retrain_schema_compatibility_ok" not in gate_failures,
    }
    packet_completeness_score = 0.0
    if packet:
        packet_completeness_score += 25.0
    if packet_complete:
        packet_completeness_score += 25.0
    if signing_material_ready:
        packet_completeness_score += 20.0
    if signature_verified:
        packet_completeness_score += 15.0
    if rollback_reference and rollback_command:
        packet_completeness_score += 15.0
    packet_completeness_score = min(round(packet_completeness_score, 2), 100.0)
    source_seed_present = bool(isinstance(packet.get("sources"), dict) and (packet.get("sources") or {}))
    repairable_packet_state = bool(
        packet
        and autopilot_state in {"assembling_packet", "awaiting_signing_material", "awaiting_signature", "repairing_readiness"}
        and (
            readiness_repair_rows
            or packet_completeness_score >= 25.0
            or bool(packet_sha256)
            or source_seed_present
        )
    )
    overall_status = "ready" if autopilot_state == "awaiting_approval" else "blocked"
    if autopilot_state == "ready_for_supervised_canary":
        overall_status = "degraded"
    if repairable_packet_state:
        overall_status = "degraded"
    recommended_actions = ordered_unique(
        [*(str(row.get("repair_hint") or "") for row in readiness_repair_rows[:3])]
        + [
            "refresh or assemble the promotion packet before any live promotion is considered" if not packet or not packet_complete else "",
            "install or refresh the promotion packet signing key so the packet verifies cleanly" if packet_complete and not signing_material_ready else "",
            "export PROMOTION_PACKET_SIGNING_KEY or stage the host key file before asking the committee packet to advance beyond assembling_packet" if not signing_material_ready else "",
            "rerun the promotion packet builder after staging signing material so the packet can move from awaiting_signature to verified" if packet_complete and signing_material_ready and not signature_verified else "",
            "keep the challenger in paper and replay until walk-forward coverage and promotion gates clear" if blockers else "",
            "record operator approval against the signed packet sha and rollback reference before promotion" if autopilot_state == "awaiting_approval" else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "autopilot_state": autopilot_state,
        "packet_complete": packet_complete,
        "signature_verified": signature_verified,
        "promotion_ready": bool(packet_complete and signature_verified and promote_ok_effective and quality_ok and (pipeline_ok or pipeline_repairable_for_canary)),
        "canary_packet_ready": canary_packet_ready,
        "approval_state": str(approval_record.get("approval_state") or ""),
        "repairable_gate_count": len(readiness_repair_rows),
        "repairable_packet_state": repairable_packet_state,
        "blocker_count": len(blockers),
        "packet_completeness_score": packet_completeness_score,
        "blockers": blockers,
        "gate_summary": {
            "promotion_readiness_ok": promote_ok_effective,
            "promotion_quality_gate_ok": quality_ok,
            "promotion_pipeline_ok": bool(pipeline_ok or pipeline_repairable_for_canary),
            "coverage_shortfall_bots": coverage_shortfall_bots,
            "raw_coverage_shortfall_bots": raw_coverage_shortfall_bots,
            "gate_failures": gate_failures,
        },
        "coverage_seed_contract": {
            "raw_coverage_shortfall_bots": raw_coverage_shortfall_bots,
            "effective_coverage_shortfall_bots": coverage_shortfall_bots,
            "seed_queue_size": coverage_seed_queue_size,
            "queued_candidate_count": queued_coverage_candidates,
            "stage_candidate_count": stage_candidate_count,
            "launch_state": gap_launch_state,
            "seed_ready": coverage_seed_ready,
            "canary_seed_ready": coverage_seed_ready and canary_packet_ready,
        },
        "committee_packet_seed_ready": committee_packet_seed_ready,
        "signed_bundle_contract": signed_bundle_contract,
        "signability_contract": signability_contract,
        "readiness_repair_contract": readiness_repair_contract,
        "evidence_bundle": evidence_bundle,
        "rollback_bundle": rollback_bundle,
        "approval_record": approval_record,
        "next_readying_actions": recommended_actions[:4],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble a promotion autopilot packet around the signed promotion evidence bundle.")
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
            "promotion_autopilot_packet "
            f"overall_status={payload.get('overall_status', '')} "
            f"autopilot_state={payload.get('autopilot_state', '')}"
        )
    return 0 if payload.get("overall_status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
