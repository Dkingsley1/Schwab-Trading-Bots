import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.feature_store_manifest as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_build_manifest_tracks_point_in_time_contract_and_lanes(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "row_count": 1200,
            "sequence_count": 18,
            "rows_path": str(tmp_path / "exports" / "training" / "runtime.jsonl"),
            "rows_sha256": "rows-hash",
            "lookback_days": 21,
            "prefer_sqlite": True,
            "coverage": {
                "top_modes": [
                    {"mode": "shadow_intraday_aggressive_equities", "row_count": 600},
                    {"mode": "shadow_dividend_equities", "row_count": 140},
                ]
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "feature_versions" / "latest.json",
        {
            "env_hash": "env-hash",
            "file_hashes": {"master_bot_registry.json": "abc"},
            "env": {"FEATURE_WINDOWS": "5,10,20"},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "snapshot_coverage_latest.json",
        {
            "coverage_ratio": 1.2,
            "min_coverage_ratio": 0.75,
            "rows_with_snapshot_id": 1200,
            "unique_snapshot_ids": 1200,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "point_in_time_event_store_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "ok": True,
            "event_count": 7,
            "category_counts": {"policy_macro": 3, "broker_readiness": 4},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "retrain_scorecard_latest.json",
        {"lineage": {"lineage_schema_version": 2}},
    )
    _write_json(
        tmp_path / "data" / "trade_history" / "trade_behavior_dataset.json",
        {
            "schema": "behavior_dataset_v3_dual_horizon",
            "feature_schema_version": "trade_behavior_features_v4",
            "horizons": {"primary_seconds": 300, "aux_seconds": 900, "blend_alpha": 0.65},
            "weights": {"neutral_horizon_disagree_downweight": 0.74},
        },
    )

    payload = src.build_manifest(tmp_path)

    assert payload["ok"] is True
    assert payload["strict_ok"] is True
    assert payload["dataset_contract"]["row_count"] == 1200
    assert payload["point_in_time_contract"]["event_count"] == 7
    assert payload["point_in_time_contract"]["complete"] is True
    assert payload["lane_partitions"][0]["lane"] == "intraday_aggressive"
    assert payload["lineage_schema_version"] == 2
    assert payload["label_contract"]["complete"] is True
    assert len(payload["contract_hashes"]["dataset_manifest_sha256"]) == 64


def test_build_manifest_falls_back_to_trade_learning_dataset_lineage_when_dual_horizon_file_is_missing(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "row_count": 600,
            "sequence_count": 9,
            "rows_path": str(tmp_path / "exports" / "training" / "runtime.jsonl"),
            "rows_sha256": "rows-hash",
            "coverage": {"top_modes": [{"mode": "shadow_intraday_aggressive_equities", "row_count": 600}]},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "snapshot_coverage_latest.json",
        {
            "coverage_ratio": 1.0,
            "min_coverage_ratio": 0.75,
            "rows_with_snapshot_id": 600,
            "unique_snapshot_ids": 600,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "point_in_time_event_store_latest.json",
        {
            "timestamp_utc": (datetime.now(timezone.utc).replace(microsecond=0)).isoformat(),
            "event_count": 3,
            "category_counts": {"broker_readiness": 3},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "retrain_scorecard_latest.json",
        {
            "lineage": {
                "lineage_schema_version": 3,
                "trade_behavior_feature_schema_version": "trade_behavior_features_v3",
                "trade_behavior_dataset_sha256": "d" * 64,
                "trade_behavior_dataset_payload_sha256": "e" * 64,
                "trade_behavior_dataset_builder_script": str(tmp_path / "scripts" / "build_trade_learning_dataset.py"),
                "trade_behavior_dataset_builder_script_sha256": "f" * 64,
            }
        },
    )
    _write_json(
        tmp_path / "data" / "trade_history" / "trade_learning_dataset.json",
        {
            "rows": 600,
            "lineage": {
                "feature_schema_version": "trade_behavior_features_v3",
                "builder_script": str(tmp_path / "scripts" / "build_trade_learning_dataset.py"),
                "builder_script_sha256": "f" * 64,
                "output_payload_sha256": "e" * 64,
            },
        },
    )

    payload = src.build_manifest(tmp_path)

    assert payload["strict_ok"] is False
    assert payload["point_in_time_contract"]["complete"] is False
    assert payload["point_in_time_contract"]["non_operational_event_categories"] == []
    assert payload["label_contract"]["contract_mode"] == "lineage_fallback"
    assert payload["label_contract"]["feature_schema_version"] == "trade_behavior_features_v3"
    assert payload["label_contract"]["source_path"].endswith("trade_learning_dataset.json")
    assert payload["evidence"]["trade_behavior_dataset"].endswith("trade_learning_dataset.json")


def test_build_manifest_marks_stale_event_store_as_not_ready(tmp_path: Path) -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    _write_json(
        tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": fresh_ts,
            "row_count": 500,
            "sequence_count": 10,
            "rows_path": str(tmp_path / "exports" / "training" / "runtime.jsonl"),
            "rows_sha256": "rows-hash",
            "coverage": {"top_modes": [{"mode": "shadow_intraday_aggressive_equities", "row_count": 500}]},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "snapshot_coverage_latest.json",
        {
            "coverage_ratio": 1.0,
            "min_coverage_ratio": 0.75,
            "rows_with_snapshot_id": 500,
            "unique_snapshot_ids": 500,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "point_in_time_event_store_latest.json",
        {
            "timestamp_utc": "2020-04-15T01:00:00+00:00",
            "ok": True,
            "event_count": 4,
            "category_counts": {"policy_macro": 2, "broker_readiness": 2},
        },
    )
    _write_json(
        tmp_path / "data" / "trade_history" / "trade_behavior_dataset.json",
        {
            "schema": "behavior_dataset_v3_dual_horizon",
            "feature_schema_version": "trade_behavior_features_v4",
            "horizons": {"primary_seconds": 300, "aux_seconds": 900, "blend_alpha": 0.65},
            "weights": {"neutral_horizon_disagree_downweight": 0.74},
        },
    )

    payload = src.build_manifest(tmp_path)

    assert payload["ok"] is True
    assert payload["strict_ok"] is False
    assert payload["point_in_time_contract"]["event_store_fresh"] is False
