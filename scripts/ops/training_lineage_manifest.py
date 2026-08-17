#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_lineage_manifest_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_latest_jsonl_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            rows = [line.strip() for line in handle if line.strip()]
    except Exception:
        return {}
    for raw in reversed(rows):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _bool(raw: Any) -> bool:
    return bool(raw)


def _first_nonempty(*values: Any) -> str:
    for raw in values:
        text = str(raw or "").strip()
        if text:
            return text
    return ""


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    experiments_root = project_root / "governance" / "experiments"
    feature_store_root = project_root / "governance" / "feature_store"

    experiment_latest = _load_latest_jsonl_row(experiments_root / "experiment_registry.jsonl")
    feature_store_manifest = _load_json(feature_store_root / "latest.json")
    replay_hash_registry = _load_json(health_root / "replay_hash_registry_guard_latest.json")
    paper_replay = _load_json(health_root / "paper_replay_drill_latest.json")
    replay_end_to_end = _load_json(health_root / "replay_end_to_end_latest.json")
    promotion_quality = _load_json(health_root / "promotion_quality_gate_latest.json")
    promotion_autopilot = _load_json(project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json")
    promotion_packet = _load_json(project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json")
    training_report = _load_json(health_root / "training_report_latest.json")
    snapshot_coverage = _load_json(health_root / "snapshot_coverage_latest.json")
    multiple_testing_guard = _load_json(project_root / "governance" / "research" / "multiple_testing_guard_latest.json")
    decay_monitor = _load_json(project_root / "governance" / "research" / "decay_monitor_latest.json")

    replayability = experiment_latest.get("replayability") if isinstance(experiment_latest.get("replayability"), dict) else {}
    packet_replayability = (
        promotion_packet.get("replayability_contract")
        if isinstance(promotion_packet.get("replayability_contract"), dict)
        else {}
    )
    packet_signature = promotion_packet.get("signature") if isinstance(promotion_packet.get("signature"), dict) else {}
    bundle_hashes = {
        "bundle_hash": _first_nonempty(replayability.get("bundle_hash"), packet_replayability.get("bundle_hash")),
        "dataset_hash": _first_nonempty(replayability.get("dataset_hash"), packet_replayability.get("dataset_hash")),
        "model_hash": _first_nonempty(replayability.get("model_hash"), packet_replayability.get("model_hash")),
        "replay_hash": _first_nonempty(replayability.get("replay_hash"), packet_replayability.get("replay_hash")),
    }
    bundle_hash_presence = {key: bool(value) for key, value in bundle_hashes.items()}
    hash_bundle_complete = all(bundle_hash_presence.values())
    packet_hash_bundle_complete = bool(
        packet_replayability.get("hash_bundle_complete", False)
        or all(
            str(packet_replayability.get(key) or "").strip()
            for key in ("dataset_hash", "model_hash", "replay_hash", "bundle_hash")
        )
    )

    dataset_contract = feature_store_manifest.get("dataset_contract") if isinstance(feature_store_manifest.get("dataset_contract"), dict) else {}
    pit_contract = feature_store_manifest.get("point_in_time_contract") if isinstance(feature_store_manifest.get("point_in_time_contract"), dict) else {}
    feature_store_lineage_ok = bool(
        feature_store_manifest.get("ok", False)
        and str(dataset_contract.get("rows_sha256") or "").strip()
        and bool(pit_contract.get("dataset_join_keys"))
    )
    replay_hash_registry_ok = bool(replay_hash_registry.get("ok", False))
    promotion_quality_ok = bool(promotion_quality.get("ok", False))
    replay_drills_ok = bool(paper_replay.get("ok", False)) and bool(replay_end_to_end.get("ok", False))
    strong_signed_packet_replay_ready = bool(
        packet_signature.get("verified", False)
        and packet_hash_bundle_complete
        and replay_hash_registry_ok
        and replay_drills_ok
    )
    exact_replay_ready = bool(
        replayability.get("exact_replay_ready", False)
        or packet_replayability.get("exact_replay_ready", False)
        or strong_signed_packet_replay_ready
    )
    idle_packet_confirmed = bool(
        packet_replayability.get("idle_scope", False)
        and bool(promotion_packet.get("packet_complete", False))
        and bool(packet_signature.get("verified", False))
    )
    training_confirmed = bool(((training_report.get("summary") or {}).get("confirmed_training_success", False)) or idle_packet_confirmed)
    snapshot_coverage_ok = bool(snapshot_coverage.get("ok", False))
    multiple_testing_failed_checks = (
        multiple_testing_guard.get("failed_checks") if isinstance(multiple_testing_guard.get("failed_checks"), list) else []
    )
    multiple_testing_contract_present = bool(
        int(multiple_testing_guard.get("family_size", 0) or 0) > 0
        and str(multiple_testing_guard.get("correction_method") or "").strip()
    )
    multiple_testing_ready = bool(
        multiple_testing_guard.get("ok", False)
        or (multiple_testing_contract_present and not multiple_testing_failed_checks)
    )
    decay_monitor_ready = bool(decay_monitor) and str(decay_monitor.get("overall_status") or "").strip().lower() not in {"", "blocked", "critical"}
    promotion_packet_seed_ready = bool(
        (
            promotion_autopilot
            and str(((promotion_autopilot.get("signed_bundle_contract") or {}).get("packet_sha256") or "")).strip()
            and int(((promotion_autopilot.get("evidence_bundle") or {}).get("source_count") or 0) or 0) > 0
        )
        or bool(promotion_packet.get("committee_packet_seed_ready", False))
    )
    promotion_packet_ready = bool(
        (promotion_autopilot and bool(promotion_autopilot.get("packet_complete", False)) and bool(promotion_autopilot.get("signature_verified", False)))
        or (bool(promotion_packet.get("packet_complete", False)) and bool(packet_signature.get("verified", False)))
        or strong_signed_packet_replay_ready
    )
    stronger_provisional_lineage_ready = bool(
        hash_bundle_complete
        and feature_store_lineage_ok
        and replay_hash_registry_ok
        and snapshot_coverage_ok
        and decay_monitor_ready
        and promotion_packet_seed_ready
    )

    lineage_contract_ready = bool(
        feature_store_lineage_ok
        and hash_bundle_complete
        and exact_replay_ready
        and replay_hash_registry_ok
    )
    promotion_bundle_ready = bool(
        lineage_contract_ready
        and promotion_quality_ok
        and training_confirmed
        and replay_drills_ok
        and snapshot_coverage_ok
        and multiple_testing_ready
        and decay_monitor_ready
        and promotion_packet_ready
    )

    missing_contracts = _ordered_unique(
        [
            "experiment_registry_row" if not experiment_latest else "",
            "feature_store_lineage" if not feature_store_lineage_ok else "",
            "bundle_hashes" if not hash_bundle_complete else "",
            "exact_replay_ready" if not exact_replay_ready else "",
            "replay_hash_registry_guard" if not replay_hash_registry_ok else "",
            "paper_replay_drill" if not bool(paper_replay.get("ok", False)) else "",
            "replay_end_to_end" if not bool(replay_end_to_end.get("ok", False)) else "",
            "snapshot_coverage" if not snapshot_coverage_ok else "",
            "multiple_testing_guard" if not multiple_testing_ready else "",
            "decay_monitor" if not decay_monitor_ready else "",
            "promotion_quality_gate" if not promotion_quality_ok else "",
            "promotion_autopilot_packet" if not promotion_packet_ready else "",
            "training_confirmation" if not training_confirmed else "",
        ]
    )

    lineage_score = 0.0
    if experiment_latest:
        lineage_score += 15.0
    if feature_store_lineage_ok:
        lineage_score += 20.0
    if bundle_hash_presence["dataset_hash"]:
        lineage_score += 10.0
    if bundle_hash_presence["model_hash"]:
        lineage_score += 10.0
    if bundle_hash_presence["replay_hash"]:
        lineage_score += 10.0
    if bundle_hash_presence["bundle_hash"]:
        lineage_score += 10.0
    if exact_replay_ready:
        lineage_score += 10.0
    if replay_hash_registry_ok:
        lineage_score += 5.0
    if replay_drills_ok:
        lineage_score += 5.0
    if snapshot_coverage_ok:
        lineage_score += 5.0
    if multiple_testing_ready:
        lineage_score += 2.5
    if decay_monitor_ready:
        lineage_score += 2.5
    if promotion_quality_ok:
        lineage_score += 5.0
    if promotion_packet_ready:
        lineage_score += 5.0
    elif promotion_packet_seed_ready:
        lineage_score += 2.5
    if strong_signed_packet_replay_ready and not promotion_packet_ready:
        lineage_score += 2.5
    if stronger_provisional_lineage_ready and not exact_replay_ready:
        lineage_score += 7.5
    lineage_score = min(round(lineage_score, 2), 100.0)

    seeded_hash_count = sum(1 for present in bundle_hash_presence.values() if present)
    repairable_lineage_contract = {
        "seeded_hash_count": seeded_hash_count,
        "experiment_present": bool(experiment_latest),
        "feature_store_lineage_ok": feature_store_lineage_ok,
        "promotion_packet_seed_ready": promotion_packet_seed_ready,
        "research_contract_ready": bool(multiple_testing_ready and decay_monitor_ready),
        "snapshot_coverage_ok": snapshot_coverage_ok,
        "stronger_provisional_lineage_ready": stronger_provisional_lineage_ready,
    }
    thin_lineage_evidence = bool(
        not experiment_latest
        or lineage_score < 45.0
        or (not feature_store_lineage_ok and seeded_hash_count < 2)
    )
    repairable_lineage_contract["lineage_recovery_ready"] = bool(
        not promotion_bundle_ready
        and not thin_lineage_evidence
        and feature_store_lineage_ok
        and lineage_score >= 65.0
        and promotion_packet_seed_ready
        and multiple_testing_ready
        and decay_monitor_ready
        and snapshot_coverage_ok
    )

    overall_status = "ready" if promotion_bundle_ready else "needs_attention"
    if thin_lineage_evidence:
        overall_status = "blocked"
    elif repairable_lineage_contract["lineage_recovery_ready"]:
        overall_status = "degraded"

    recommended_actions = _ordered_unique(
        [
            "record dataset/model/replay/bundle hashes for every training candidate before promotion review"
            if not hash_bundle_complete
            else "",
            "repair feature-store lineage so point-in-time joins and dataset row hashes stay explicit"
            if not feature_store_lineage_ok
            else "",
            "keep replay hash registry healthy before trusting immutable experiment lineage"
            if not replay_hash_registry_ok
            else "",
            "refresh paper and end-to-end replay drills so lineage proof is backed by working replay surfaces"
            if not replay_drills_ok
            else "",
            "publish snapshot coverage before claiming promotion-ready lineage"
            if not snapshot_coverage_ok
            else "",
            "record multiple-testing control and decay-monitor artifacts before treating the promotion bundle as institutional-grade"
            if not (multiple_testing_ready and decay_monitor_ready)
            else "",
            "finish the signed promotion packet before treating the lineage bundle as promotion-ready"
            if not promotion_packet_ready
            else "",
            "wait for confirmed training success and promotion quality gate before promoting a lineage-complete candidate"
            if lineage_contract_ready and not promotion_bundle_ready
            else "",
        ]
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "lineage_score": lineage_score,
        "latest_experiment_id": str(experiment_latest.get("experiment_id") or "").strip(),
        "bundle_hashes": bundle_hashes,
        "bundle_hash_presence": bundle_hash_presence,
        "hash_bundle_complete": hash_bundle_complete,
        "exact_replay_ready": exact_replay_ready,
        "strong_signed_packet_replay_ready": strong_signed_packet_replay_ready,
        "stronger_provisional_lineage_ready": stronger_provisional_lineage_ready,
        "feature_store_lineage_ok": feature_store_lineage_ok,
        "feature_store_schema_version": int(feature_store_manifest.get("lineage_schema_version", 0) or 0),
        "replay_hash_registry_ok": replay_hash_registry_ok,
        "replay_drills_ok": replay_drills_ok,
        "snapshot_coverage_ok": snapshot_coverage_ok,
        "multiple_testing_ready": multiple_testing_ready,
        "multiple_testing_contract_present": multiple_testing_contract_present,
        "decay_monitor_ready": decay_monitor_ready,
        "training_confirmed": training_confirmed,
        "promotion_quality_ok": promotion_quality_ok,
        "promotion_packet_seed_ready": promotion_packet_seed_ready,
        "promotion_packet_ready": promotion_packet_ready,
        "repairable_lineage_contract": repairable_lineage_contract,
        "lineage_contract_ready": lineage_contract_ready,
        "promotion_bundle_ready": promotion_bundle_ready,
        "missing_contracts": missing_contracts,
        "recommended_actions": recommended_actions,
        "source_artifacts": {
            "experiment_registry": str(experiments_root / "experiment_registry.jsonl"),
            "feature_store_manifest": str(feature_store_root / "latest.json"),
            "replay_hash_registry_guard": str(health_root / "replay_hash_registry_guard_latest.json"),
            "paper_replay_drill": str(health_root / "paper_replay_drill_latest.json"),
            "replay_end_to_end": str(health_root / "replay_end_to_end_latest.json"),
            "promotion_quality_gate": str(health_root / "promotion_quality_gate_latest.json"),
            "promotion_packet": str(project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json"),
            "promotion_autopilot_packet": str(project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json"),
            "training_report": str(health_root / "training_report_latest.json"),
            "snapshot_coverage": str(health_root / "snapshot_coverage_latest.json"),
            "multiple_testing_guard": str(project_root / "governance" / "research" / "multiple_testing_guard_latest.json"),
            "decay_monitor": str(project_root / "governance" / "research" / "decay_monitor_latest.json"),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish an immutable training-lineage contract for dataset/model/replay hashes and promotion proof.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "training_lineage_manifest "
            f"overall_status={payload.get('overall_status', '')} "
            f"lineage_score={float(payload.get('lineage_score', 0.0) or 0.0):.2f}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
