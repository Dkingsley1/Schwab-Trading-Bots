import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.retrain_schema_compatibility_guard as src


def test_schema_compatibility_allows_lineage_schema_population_from_zero_baseline() -> None:
    payload = src.build_payload(
        feature_store_manifest={
            "strict_ok": True,
            "lineage_schema_version": 1,
            "point_in_time_contract": {
                "complete": True,
                "dataset_join_keys": ["snapshot_id", "symbol", "mode", "timestamp_utc"],
                "event_join_keys": ["join_key", "category", "timestamp_utc"],
            },
            "label_contract": {
                "feature_schema_version": "trade_behavior_features_v3",
                "horizons": {"primary_seconds": 0, "aux_seconds": 0},
            },
        },
        promotion_packet={
            "dataset": {
                "dataset_join_keys": ["snapshot_id", "symbol", "mode", "timestamp_utc"],
                "event_join_keys": ["join_key", "category", "timestamp_utc"],
                "feature_schema_version": "trade_behavior_features_v3",
                "label_horizons": {"primary_seconds": 0, "aux_seconds": 0},
                "lineage_schema_version": 0,
            }
        },
        schema_migration_guard={"ok": True},
    )

    assert payload["ok"] is True
    assert payload["failed_checks"] == []
    assert payload["drifted_fields"] == []
