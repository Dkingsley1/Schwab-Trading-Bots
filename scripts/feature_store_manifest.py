#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "feature_store" / "latest.json"

_BASE_LOOKBACK_BY_LANE = {
    "intraday_aggressive": 30,
    "aggressive": 30,
    "swing_aggressive": 45,
    "crypto": 21,
    "crypto_futures": 21,
    "fx": 30,
    "futures": 30,
    "conservative": 45,
    "dividend": 60,
    "bond": 60,
    "paper": 14,
    "other": 30,
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_first_json(paths: list[Path]) -> tuple[dict[str, Any], Path]:
    for path in paths:
        payload = _load_json(path)
        if payload:
            return payload, path
    return {}, (paths[0] if paths else Path())


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


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _artifact_freshness(raw_ts: Any, *, now: datetime, max_age_hours: float) -> dict[str, Any]:
    ts = _parse_ts(raw_ts)
    if ts is None:
        return {"fresh": False, "age_hours": None, "timestamp_utc": ""}
    age_hours = max((now - ts).total_seconds() / 3600.0, 0.0)
    return {
        "fresh": bool(age_hours <= max(float(max_age_hours), 0.25)),
        "age_hours": round(age_hours, 6),
        "timestamp_utc": ts.isoformat(),
    }


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path_text: Any) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mode_to_lane(mode: str) -> str:
    text = str(mode or "").strip().lower()
    if "crypto_futures" in text:
        return "crypto_futures"
    if "crypto" in text:
        return "crypto"
    if "intraday" in text:
        return "intraday_aggressive"
    if "swing" in text:
        return "swing_aggressive"
    if "dividend" in text:
        return "dividend"
    if "bond" in text:
        return "bond"
    if "fx" in text:
        return "fx"
    if "futures" in text:
        return "futures"
    if "conservative" in text:
        return "conservative"
    if "aggressive" in text:
        return "aggressive"
    if "paper" in text:
        return "paper"
    return "other"


def _lane_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = snapshot.get("coverage") if isinstance(snapshot.get("coverage"), dict) else {}
    top_modes = coverage.get("top_modes") if isinstance(coverage.get("top_modes"), list) else []
    lane_totals: dict[str, int] = {}
    for row in top_modes:
        if not isinstance(row, dict):
            continue
        lane = _mode_to_lane(str(row.get("mode") or ""))
        lane_totals[lane] = lane_totals.get(lane, 0) + _safe_int(row.get("row_count"), 0)
    rows = [{"lane": lane, "row_count": row_count} for lane, row_count in lane_totals.items() if row_count > 0]
    rows.sort(key=lambda item: (-int(item["row_count"]), str(item["lane"])))
    return rows


def _recommended_lookback_days(lane: str, row_count: int, total_rows: int) -> int:
    total = max(int(total_rows), 1)
    share = _safe_float(row_count, 0.0) / total
    base = int(_BASE_LOOKBACK_BY_LANE.get(lane, _BASE_LOOKBACK_BY_LANE["other"]))
    if share >= 0.30:
        return max(base - 7, 14)
    if share <= 0.05:
        return min(base + 15, 90)
    return base


def build_manifest(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"

    snapshot = _load_json(health_root / "runtime_training_snapshot_latest.json")
    feature_versions_path = project_root / "governance" / "feature_versions" / "latest.json"
    feature_versions = _load_json(feature_versions_path)
    prior_manifest = _load_json(project_root / "governance" / "feature_store" / "latest.json")
    coverage = _load_json(health_root / "snapshot_coverage_latest.json")
    event_store = _load_json(health_root / "point_in_time_event_store_latest.json")
    retrain_scorecard = _load_json(health_root / "retrain_scorecard_latest.json")
    trade_behavior_dataset, trade_behavior_dataset_path = _load_first_json(
        [
            project_root / "data" / "trade_history" / "trade_behavior_dataset.json",
            project_root / "data" / "trade_history" / "trade_learning_dataset.json",
        ]
    )

    row_count = _safe_int(snapshot.get("row_count"), 0)
    sequence_count = _safe_int(snapshot.get("sequence_count"), 0)
    rows_path = str(snapshot.get("rows_path") or "")
    rows_sha256 = str(snapshot.get("rows_sha256") or "")
    coverage_ratio = _safe_float(coverage.get("coverage_ratio"), 0.0)
    min_coverage_ratio = _safe_float(coverage.get("min_coverage_ratio"), 0.0)
    event_count = _safe_int(event_store.get("event_count"), 0)
    lineage = retrain_scorecard.get("lineage") if isinstance(retrain_scorecard.get("lineage"), dict) else {}
    prior_feature_contract = (
        prior_manifest.get("feature_contract") if isinstance(prior_manifest.get("feature_contract"), dict) else {}
    )
    prior_label_contract = (
        prior_manifest.get("label_contract") if isinstance(prior_manifest.get("label_contract"), dict) else {}
    )
    dataset_lineage = trade_behavior_dataset.get("lineage") if isinstance(trade_behavior_dataset.get("lineage"), dict) else {}
    lane_rows = _lane_rows(snapshot)
    category_counts = event_store.get("category_counts") if isinstance(event_store.get("category_counts"), dict) else {}
    event_categories = sorted(str(key) for key in category_counts.keys())
    non_operational_event_categories = [key for key in event_categories if key not in {"broker_readiness"}]
    snapshot_freshness = _artifact_freshness(snapshot.get("timestamp_utc"), now=now, max_age_hours=12.0)
    event_store_freshness = _artifact_freshness(event_store.get("timestamp_utc"), now=now, max_age_hours=6.0)
    file_hashes = feature_versions.get("file_hashes") if isinstance(feature_versions.get("file_hashes"), dict) else {}
    if not file_hashes:
        file_hashes = (
            prior_feature_contract.get("tracked_file_hashes")
            if isinstance(prior_feature_contract.get("tracked_file_hashes"), dict)
            else {}
        )
    horizons = trade_behavior_dataset.get("horizons") if isinstance(trade_behavior_dataset.get("horizons"), dict) else {}
    if not horizons:
        horizons = prior_label_contract.get("horizons") if isinstance(prior_label_contract.get("horizons"), dict) else {}
    label_weights = trade_behavior_dataset.get("weights") if isinstance(trade_behavior_dataset.get("weights"), dict) else {}
    feature_schema_version = str(
        trade_behavior_dataset.get("feature_schema_version")
        or dataset_lineage.get("feature_schema_version")
        or lineage.get("trade_behavior_feature_schema_version")
        or prior_label_contract.get("feature_schema_version")
        or ""
    )
    dataset_schema = str(
        trade_behavior_dataset.get("schema")
        or trade_behavior_dataset.get("dataset_schema")
        or prior_label_contract.get("schema")
        or ("trade_learning_dataset" if trade_behavior_dataset_path.name == "trade_learning_dataset.json" and trade_behavior_dataset else "")
    )
    trade_behavior_dataset_path_text = (
        str(trade_behavior_dataset_path)
        if trade_behavior_dataset and trade_behavior_dataset_path
        else str(project_root / "data" / "trade_history" / "trade_behavior_dataset.json")
    )
    auto_source_paths = {
        "runtime_training_rows": rows_path,
        "runtime_training_snapshot": str(health_root / "runtime_training_snapshot_latest.json"),
        "point_in_time_event_store": str(health_root / "point_in_time_event_store_latest.json"),
        "trade_behavior_dataset": trade_behavior_dataset_path_text,
        "feature_versions": str(feature_versions_path),
    }
    auto_file_hashes = {
        name: digest
        for name, digest in (
            (name, _sha256_file(path_text))
            for name, path_text in auto_source_paths.items()
        )
        if digest
    }
    tracked_file_hashes = dict(file_hashes) if file_hashes else dict(auto_file_hashes)
    tracked_file_hash_source = "feature_versions" if file_hashes else "auto_discovered_sources"
    if not tracked_file_hashes and isinstance(prior_feature_contract.get("tracked_file_hashes"), dict):
        tracked_file_hashes = dict(prior_feature_contract.get("tracked_file_hashes") or {})
        tracked_file_hash_source = "feature_store_fallback"

    lane_partitions = []
    for row in lane_rows[:16]:
        lane = str(row.get("lane") or "")
        lane_count = _safe_int(row.get("row_count"), 0)
        lane_partitions.append(
            {
                "lane": lane,
                "row_count": lane_count,
                "share": round(lane_count / max(row_count, 1), 6),
                "partition_key": f"lane={lane}",
                "recommended_lookback_days": _recommended_lookback_days(lane, lane_count, row_count),
            }
        )

    datasets = [
        {
            "name": "runtime_training_snapshot",
            "path": rows_path,
            "row_count": row_count,
            "sha256": rows_sha256,
            "point_in_time_key": "timestamp_utc",
            "join_keys": ["snapshot_id", "symbol", "mode"],
            "availability_contract": "append_only_snapshot",
        },
        {
            "name": "event_store",
            "path": str(health_root / "point_in_time_event_store_latest.json"),
            "row_count": event_count,
            "sha256": "",
            "point_in_time_key": "timestamp_utc",
            "join_keys": ["join_key", "category"],
            "availability_contract": "recent_event_window",
        },
    ]
    if trade_behavior_dataset:
        datasets.append(
            {
                "name": "trade_behavior_dataset",
                "path": trade_behavior_dataset_path_text,
                "row_count": _safe_int(trade_behavior_dataset.get("rows"), 0),
                "sha256": str(lineage.get("trade_behavior_dataset_sha256") or _sha256_file(trade_behavior_dataset_path_text)),
                "point_in_time_key": "timestamp_utc",
                "join_keys": ["feature_schema_version"],
                "availability_contract": "content_addressed_dataset_payload",
            }
        )

    point_in_time_contract = {
        "effective_time_key": "timestamp_utc",
        "dataset_join_keys": ["snapshot_id", "symbol", "mode", "timestamp_utc"],
        "event_join_keys": ["join_key", "category", "timestamp_utc"],
        "snapshot_coverage_ratio": round(coverage_ratio, 6),
        "snapshot_coverage_floor": round(min_coverage_ratio, 6),
        "rows_with_snapshot_id": _safe_int(coverage.get("rows_with_snapshot_id"), 0),
        "unique_snapshot_ids": _safe_int(coverage.get("unique_snapshot_ids"), 0),
        "event_count": event_count,
        "event_categories": event_categories,
        "event_category_count": len(event_categories),
        "non_operational_event_categories": non_operational_event_categories,
        "event_store_fresh": bool(event_store_freshness["fresh"]),
        "event_store_age_hours": event_store_freshness["age_hours"],
    }
    event_contract_ready = bool(
        event_count > 0
        and bool(event_categories)
        and (len(event_categories) >= 2 or bool(non_operational_event_categories))
        and bool(non_operational_event_categories)
    )
    point_in_time_complete = bool(
        row_count > 0
        and rows_sha256
        and bool(point_in_time_contract["dataset_join_keys"])
        and bool(point_in_time_contract["event_join_keys"])
        and coverage_ratio >= min_coverage_ratio
        and event_contract_ready
        and bool(event_store_freshness["fresh"])
        and bool(tracked_file_hashes)
    )
    point_in_time_contract["complete"] = point_in_time_complete

    dataset_payload_sha256 = str(
        dataset_lineage.get("output_payload_sha256")
        or lineage.get("trade_behavior_dataset_payload_sha256")
        or ""
    )
    dataset_builder_script = str(
        dataset_lineage.get("builder_script")
        or lineage.get("trade_behavior_dataset_builder_script")
        or ""
    )
    dataset_builder_script_sha256 = str(
        dataset_lineage.get("builder_script_sha256")
        or lineage.get("trade_behavior_dataset_builder_script_sha256")
        or ""
    )
    dataset_sha256 = str(
        lineage.get("trade_behavior_dataset_sha256")
        or _sha256_file(trade_behavior_dataset_path_text)
    )
    label_lineage_complete = bool(
        trade_behavior_dataset
        and trade_behavior_dataset_path_text
        and (dataset_payload_sha256 or dataset_sha256)
        and (dataset_builder_script_sha256 or dataset_builder_script)
    )
    label_contract = {
        "schema": dataset_schema,
        "feature_schema_version": feature_schema_version,
        "source_path": trade_behavior_dataset_path_text,
        "metadata_source": trade_behavior_dataset_path.name if trade_behavior_dataset_path_text else "",
        "horizons": {
            "primary_seconds": _safe_int(horizons.get("primary_seconds"), 0),
            "aux_seconds": _safe_int(horizons.get("aux_seconds"), 0),
            "blend_alpha": _safe_float(horizons.get("blend_alpha"), 0.0),
        },
        "weighting": {
            "neutral_horizon_disagree_downweight": _safe_float(
                label_weights.get("neutral_horizon_disagree_downweight"),
                0.0,
            ),
        },
        "lineage": {
            "dataset_sha256": dataset_sha256,
            "payload_sha256": dataset_payload_sha256,
            "builder_script": dataset_builder_script,
            "builder_script_sha256": dataset_builder_script_sha256,
        },
    }
    label_contract["contract_mode"] = (
        "explicit_horizon"
        if int(label_contract["horizons"]["primary_seconds"]) > 0
        else ("lineage_fallback" if label_lineage_complete else "incomplete")
    )
    label_contract["complete"] = bool(
        label_contract["feature_schema_version"]
        and (
            int(label_contract["horizons"]["primary_seconds"]) > 0
            or label_lineage_complete
        )
    )
    env_payload = feature_versions.get("env") if isinstance(feature_versions.get("env"), dict) else {}
    if not env_payload:
        env_payload = prior_feature_contract.get("env") if isinstance(prior_feature_contract.get("env"), dict) else {}
    env_hash = str(feature_versions.get("env_hash") or "").strip()
    env_hash_source = "feature_versions"
    if not env_hash:
        env_hash = str(prior_feature_contract.get("env_hash") or "").strip()
        env_hash_source = "feature_store_fallback" if env_hash else "manifest_fallback"
    if not env_hash:
        env_hash = _sha256_json(
            {
                "env": env_payload,
                "tracked_file_hashes": tracked_file_hashes,
                "feature_schema_version": feature_schema_version,
            }
        )
        env_hash_source = "manifest_fallback"

    feature_contract = {
        "env_hash": env_hash,
        "env_hash_source": env_hash_source,
        "tracked_file_hash_count": len(tracked_file_hashes),
        "tracked_files": sorted(str(key) for key in tracked_file_hashes.keys()),
        "tracked_file_hashes": tracked_file_hashes,
        "tracked_file_hash_source": tracked_file_hash_source,
        "env": env_payload,
    }

    dataset_contract = {
        "rows_path": rows_path,
        "row_count": row_count,
        "sequence_count": sequence_count,
        "rows_sha256": rows_sha256,
        "lookback_days": _safe_int(snapshot.get("lookback_days"), 0),
        "prefer_sqlite": bool(snapshot.get("prefer_sqlite", False)),
        "mode_allowlist": snapshot.get("mode_allowlist") if isinstance(snapshot.get("mode_allowlist"), list) else [],
        "symbol_allowlist": snapshot.get("symbol_allowlist") if isinstance(snapshot.get("symbol_allowlist"), list) else [],
        "snapshot_fresh": bool(snapshot_freshness["fresh"]),
        "snapshot_age_hours": snapshot_freshness["age_hours"],
    }

    contract_hashes = {
        "dataset_contract_sha256": _sha256_json(dataset_contract),
        "point_in_time_contract_sha256": _sha256_json(point_in_time_contract),
        "feature_contract_sha256": _sha256_json(feature_contract),
        "label_contract_sha256": _sha256_json(label_contract),
    }
    contract_hashes["dataset_manifest_sha256"] = _sha256_json(
        {
            "dataset_contract_sha256": contract_hashes["dataset_contract_sha256"],
            "point_in_time_contract_sha256": contract_hashes["point_in_time_contract_sha256"],
            "feature_contract_sha256": contract_hashes["feature_contract_sha256"],
            "label_contract_sha256": contract_hashes["label_contract_sha256"],
        }
    )

    ok = bool(
        row_count > 0
        and sequence_count > 0
        and rows_path
        and rows_sha256
        and bool(tracked_file_hashes)
        and coverage_ratio >= min_coverage_ratio
        and bool(snapshot_freshness["fresh"])
    )
    strict_ok = bool(ok and point_in_time_complete and bool(label_contract.get("complete", False)))
    overall_status = "ready" if ok else "needs_work"
    if row_count <= 0 or not rows_sha256:
        overall_status = "blocked"

    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "strict_status": "ready" if strict_ok else ("blocked" if overall_status == "blocked" else "needs_work"),
        "manifest_version": 1,
        "lineage_schema_version": _safe_int(lineage.get("lineage_schema_version"), 0),
        "strict_ok": strict_ok,
        "dataset_contract": dataset_contract,
        "point_in_time_contract": point_in_time_contract,
        "feature_contract": feature_contract,
        "label_contract": label_contract,
        "contract_hashes": contract_hashes,
        "lane_partitions": lane_partitions,
        "datasets": datasets,
        "evidence": {
            "runtime_training_snapshot": str(health_root / "runtime_training_snapshot_latest.json"),
            "feature_versions": str(feature_versions_path),
            "snapshot_coverage": str(health_root / "snapshot_coverage_latest.json"),
            "point_in_time_event_store": str(health_root / "point_in_time_event_store_latest.json"),
            "retrain_scorecard": str(health_root / "retrain_scorecard_latest.json"),
            "trade_behavior_dataset": trade_behavior_dataset_path_text,
        },
        "auto_source_hashes": auto_file_hashes,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a canonical feature-store manifest from runtime lineage artifacts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_manifest(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "feature_store_manifest "
            f"status={payload['overall_status']} "
            f"rows={int(((payload.get('dataset_contract') or {}).get('row_count', 0) or 0))} "
            f"lanes={len(payload.get('lane_partitions', []))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
