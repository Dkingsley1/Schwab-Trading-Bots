#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "retrain_schema_compatibility_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _signature_payload(feature_store_manifest: dict[str, Any]) -> dict[str, Any]:
    point_in_time = feature_store_manifest.get("point_in_time_contract") if isinstance(feature_store_manifest.get("point_in_time_contract"), dict) else {}
    label_contract = feature_store_manifest.get("label_contract") if isinstance(feature_store_manifest.get("label_contract"), dict) else {}
    horizons = label_contract.get("horizons") if isinstance(label_contract.get("horizons"), dict) else {}
    return {
        "dataset_join_keys": _normalize_str_list(point_in_time.get("dataset_join_keys")),
        "event_join_keys": _normalize_str_list(point_in_time.get("event_join_keys")),
        "feature_schema_version": str(label_contract.get("feature_schema_version") or "").strip(),
        "label_primary_seconds": _safe_int(horizons.get("primary_seconds"), 0),
        "label_aux_seconds": _safe_int(horizons.get("aux_seconds"), 0),
        "lineage_schema_version": _safe_int(feature_store_manifest.get("lineage_schema_version"), 0),
    }


def _baseline_signature(promotion_packet: dict[str, Any]) -> dict[str, Any]:
    dataset = promotion_packet.get("dataset") if isinstance(promotion_packet.get("dataset"), dict) else {}
    horizons = dataset.get("label_horizons") if isinstance(dataset.get("label_horizons"), dict) else {}
    return {
        "dataset_join_keys": _normalize_str_list(dataset.get("dataset_join_keys")),
        "event_join_keys": _normalize_str_list(dataset.get("event_join_keys")),
        "feature_schema_version": str(dataset.get("feature_schema_version") or "").strip(),
        "label_primary_seconds": _safe_int(horizons.get("primary_seconds"), 0),
        "label_aux_seconds": _safe_int(horizons.get("aux_seconds"), 0),
        "lineage_schema_version": _safe_int(dataset.get("lineage_schema_version"), 0),
    }


def _signature_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _compatible_lineage_schema_upgrade(current_signature: dict[str, Any], baseline_signature: dict[str, Any]) -> bool:
    return (
        _safe_int(baseline_signature.get("lineage_schema_version"), 0) <= 0
        and _safe_int(current_signature.get("lineage_schema_version"), 0) > 0
    )


def build_payload(
    *,
    feature_store_manifest: dict[str, Any],
    promotion_packet: dict[str, Any],
    schema_migration_guard: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    current_signature = _signature_payload(feature_store_manifest)
    baseline_signature = _baseline_signature(promotion_packet)
    point_in_time = (
        feature_store_manifest.get("point_in_time_contract")
        if isinstance(feature_store_manifest.get("point_in_time_contract"), dict)
        else {}
    )
    feature_store_strict_ok = bool(feature_store_manifest.get("strict_ok", False))
    feature_store_seed_ready = bool(
        feature_store_strict_ok
        or feature_store_manifest.get("strict_seed_ready", False)
    )
    point_in_time_complete = bool(point_in_time.get("complete", False))
    point_in_time_seed_ready = bool(
        point_in_time_complete
        or point_in_time.get("seed_ready", False)
    )
    schema_migration_ok = bool(schema_migration_guard.get("ok", False))
    baseline_ready = any(
        bool(value)
        for value in [
            baseline_signature["dataset_join_keys"],
            baseline_signature["event_join_keys"],
            baseline_signature["feature_schema_version"],
            baseline_signature["label_primary_seconds"],
        ]
    )
    drifted_fields: list[str] = []
    for key in current_signature:
        if not baseline_ready:
            continue
        if key == "lineage_schema_version" and _compatible_lineage_schema_upgrade(current_signature, baseline_signature):
            continue
        if current_signature.get(key) != baseline_signature.get(key):
            drifted_fields.append(key)

    failed_checks: list[str] = []
    if not feature_store_seed_ready:
        failed_checks.append("feature_store_manifest_not_strict_ready")
    if not point_in_time_seed_ready:
        failed_checks.append("point_in_time_contract_incomplete")
    migration_required = bool(drifted_fields)
    if migration_required and schema_migration_guard and not schema_migration_ok:
        failed_checks.append("schema_migration_guard_not_ready")
    if baseline_ready and drifted_fields:
        failed_checks.append("schema_signature_drift")

    compatibility_seed_ready = bool(
        baseline_ready
        and feature_store_seed_ready
        and point_in_time_seed_ready
        and not drifted_fields
    )
    if not baseline_ready:
        overall_status = "warmup"
    elif not failed_checks and compatibility_seed_ready and not (
        feature_store_strict_ok and point_in_time_complete and (schema_migration_ok or not migration_required)
    ):
        overall_status = "degraded"
    else:
        overall_status = "ready" if not failed_checks else "blocked"

    compatibility_score = 0.0
    if baseline_ready:
        compatibility_score += 25.0
    if feature_store_seed_ready:
        compatibility_score += 20.0
    if point_in_time_seed_ready:
        compatibility_score += 20.0
    if not drifted_fields:
        compatibility_score += 20.0
    if schema_migration_ok or not migration_required:
        compatibility_score += 15.0
    compatibility_score = min(round(compatibility_score, 2), 100.0)

    summary = (
        "schema-compatible with seeded lineage evidence; migration manifest can stay advisory until drift appears"
        if compatibility_seed_ready and overall_status == "degraded"
        else (
            "schema signatures are aligned and retrain contract is ready"
            if not failed_checks and baseline_ready
            else (
                "waiting for a promotion baseline before schema-sensitive retrains can be evaluated"
                if not baseline_ready
                else "schema-sensitive retrains remain blocked until lineage or migration evidence is repaired"
            )
        )
    )
    recommended_actions: list[str] = []
    if not feature_store_seed_ready:
        recommended_actions.append("refresh the feature-store manifest so dataset and point-in-time lineage evidence is available")
    if not point_in_time_seed_ready:
        recommended_actions.append("repair the point-in-time contract before allowing schema-sensitive retrains")
    if migration_required and not schema_migration_ok:
        recommended_actions.append("refresh the schema migration manifest before promoting a signature drift")
    if drifted_fields:
        recommended_actions.append("rebuild the promotion baseline after the schema change is approved")
    elif compatibility_seed_ready and overall_status == "degraded":
        recommended_actions.append("promote the seeded lineage evidence to a fully strict-ready contract when the event-backed point-in-time store returns")

    return {
        "timestamp_utc": now,
        "schema_version": 1,
        "ok": bool(baseline_ready and not failed_checks),
        "overall_status": overall_status,
        "baseline_ready": baseline_ready,
        "failed_checks": failed_checks,
        "drifted_fields": drifted_fields,
        "migration_required": migration_required,
        "feature_store_seed_ready": feature_store_seed_ready,
        "point_in_time_seed_ready": point_in_time_seed_ready,
        "compatibility_seed_ready": compatibility_seed_ready,
        "compatibility_score": compatibility_score,
        "summary": summary,
        "current_signature": {
            **current_signature,
            "schema_signature_sha256": _signature_hash(current_signature),
        },
        "baseline_signature": {
            **baseline_signature,
            "schema_signature_sha256": _signature_hash(baseline_signature) if baseline_ready else "",
        },
        "top_actions": recommended_actions,
        "recommended_actions": recommended_actions,
        "source_artifacts": {
            "feature_store_manifest": str(PROJECT_ROOT / "governance" / "feature_store" / "latest.json"),
            "promotion_packet": str(PROJECT_ROOT / "governance" / "champion_challenger" / "promotion_packet_latest.json"),
            "schema_migration_guard": str(PROJECT_ROOT / "governance" / "migrations" / "latest.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail fast on feature or label schema drift before retrain.")
    parser.add_argument("--feature-store-manifest", default=str(PROJECT_ROOT / "governance" / "feature_store" / "latest.json"))
    parser.add_argument("--promotion-packet-file", default=str(PROJECT_ROOT / "governance" / "champion_challenger" / "promotion_packet_latest.json"))
    parser.add_argument("--schema-migration-file", default=str(PROJECT_ROOT / "governance" / "migrations" / "latest.json"))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        feature_store_manifest=_load_json(Path(args.feature_store_manifest)),
        promotion_packet=_load_json(Path(args.promotion_packet_file)),
        schema_migration_guard=_load_json(Path(args.schema_migration_file)),
    )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "retrain_schema_compatibility_guard "
            f"ok={str(payload['ok']).lower()} "
            f"status={payload.get('overall_status', 'unknown')} "
            f"drifted={len(payload.get('drifted_fields', []))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
