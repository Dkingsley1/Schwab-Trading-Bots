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

    point_in_time = feature_store_manifest.get("point_in_time_contract") if isinstance(feature_store_manifest.get("point_in_time_contract"), dict) else {}
    failed_checks: list[str] = []
    if not bool(feature_store_manifest.get("strict_ok", False)):
        failed_checks.append("feature_store_manifest_not_strict_ready")
    if not bool(point_in_time.get("complete", False)):
        failed_checks.append("point_in_time_contract_incomplete")
    if schema_migration_guard and not bool(schema_migration_guard.get("ok", False)):
        failed_checks.append("schema_migration_guard_not_ready")
    if baseline_ready and drifted_fields:
        failed_checks.append("schema_signature_drift")

    if not baseline_ready:
        overall_status = "warmup"
    else:
        overall_status = "ready" if not failed_checks else "blocked"

    return {
        "timestamp_utc": now,
        "schema_version": 1,
        "ok": not failed_checks,
        "overall_status": overall_status,
        "baseline_ready": baseline_ready,
        "failed_checks": failed_checks,
        "drifted_fields": drifted_fields,
        "current_signature": {
            **current_signature,
            "schema_signature_sha256": _signature_hash(current_signature),
        },
        "baseline_signature": {
            **baseline_signature,
            "schema_signature_sha256": _signature_hash(baseline_signature) if baseline_ready else "",
        },
        "top_actions": (
            ["refresh the migration manifest and promotion baseline before allowing schema-sensitive retrains"]
            if failed_checks
            else []
        ),
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
