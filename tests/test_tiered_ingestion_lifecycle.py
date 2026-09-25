import json
from pathlib import Path

import pytest

from core.tiered_ingestion_lifecycle import (
    load_lifecycle_policy,
    plan_tiered_ingestion_lifecycle,
)


def _policy() -> dict:
    return {
        "policy_id": "test_policy",
        "policy_sha256": "a" * 64,
        "thresholds": {
            "target_segment_bytes": 100,
            "small_segment_max_bytes": 40,
            "min_compaction_inputs": 2,
            "max_compaction_inputs": 4,
            "max_compaction_wave_bytes": 200,
            "sealed_min_age_days": 1,
            "cold_after_days": 14,
            "small_segment_soft_count": 3,
            "small_segment_hard_count": 8,
            "minimum_hot_free_bytes": 1000,
            "critical_hot_free_bytes": 500,
            "snapshot_retention_count": 10,
            "orphan_grace_hours": 72,
        },
        "generic_compaction_roles": ["governance_telemetry"],
        "generic_compaction_classifications": ["below_manifest_candidate_floor"],
        "protected_classifications": [
            "current_day_hot_path_hold",
            "keep_hot_critical",
            "stateful_sql_compaction_only",
        ],
        "design_sources": [],
    }


def _entry(path: str, **overrides) -> dict:
    row = {
        "relative_path": path,
        "size_bytes": 25,
        "family": "governance_events",
        "service_role": "governance_telemetry",
        "classification": "below_manifest_candidate_floor",
        "age_days": 2,
        "source_stat_fingerprint": {"size_bytes": 25, "mtime_ns": 1, "inode": 2},
    }
    row.update(overrides)
    return row


def test_planner_compacts_only_sealed_small_segments_and_tiers_completed_files() -> None:
    entries = [
        _entry(f"governance/events/event_2026070{day}.jsonl") for day in range(1, 5)
    ]
    entries.extend(
        [
            _entry(
                "governance/events/event_20260821.jsonl",
                classification="current_day_hot_path_hold",
                age_days=0,
            ),
            _entry(
                "decisions/trade_decisions_20260820.jsonl",
                classification="keep_hot_critical",
                service_role="live_decisioning",
            ),
            _entry(
                "decision_explanations/archive_20260701.jsonl",
                size_bytes=250,
                family="decision_explanations",
                service_role="explainability_archive",
                classification="eligible_manifest_backed_offload",
                age_days=30,
            ),
            _entry(
                "data/sql_link_shards/state.sqlite3",
                size_bytes=500,
                family="sql_link_shards",
                service_role="stateful_sql",
                classification="stateful_sql_compaction_only",
                age_days=None,
            ),
        ]
    )

    plan = plan_tiered_ingestion_lifecycle(
        entries,
        policy=_policy(),
        available_hot_bytes=5000,
    )

    assert plan["operational_status"] == "ready"
    assert plan["compaction_plan"]["group_count"] == 1
    assert plan["compaction_plan"]["input_count"] == 4
    assert plan["tiering_plan"]["cold_candidate_count"] == 1
    assert plan["runtime_evidence"]["protected_entry_count"] == 3
    assert plan["stateful_sql_plan"]["file_count"] == 1
    assert plan["execution_authority"] is False
    assert plan["source_delete_authority"] is False


def test_planner_reports_backpressure_without_granting_throttle_authority() -> None:
    entries = [_entry(f"governance/events/event_202607{day:02d}.jsonl") for day in range(1, 9)]

    plan = plan_tiered_ingestion_lifecycle(
        entries,
        policy=_policy(),
        available_hot_bytes=400,
        hot_path_over_budget_bytes=50,
    )

    assert plan["work_state"] == "intake_throttle_advisory"
    assert plan["backpressure_contract"]["intake_throttle_authority"] is False
    assert plan["compaction_plan"]["input_bytes"] <= 200


def test_retirement_evidence_only_creates_an_operator_review_candidate() -> None:
    entry = _entry(
        "decision_explanations/archive_20260701.jsonl",
        size_bytes=250,
        family="decision_explanations",
        service_role="explainability_archive",
        classification="eligible_manifest_backed_offload",
        age_days=30,
        retirement_evidence={
            "verified_cold_copy": True,
            "sha256_match": True,
            "restore_probe": True,
            "retention_gate": True,
            "snapshot_reference_count": 0,
            "orphan_age_hours": 96,
        },
    )

    plan = plan_tiered_ingestion_lifecycle(
        [entry],
        policy=_policy(),
        available_hot_bytes=5000,
    )

    assert plan["retirement_gate"]["review_ready_count"] == 1
    assert plan["retirement_gate"]["source_delete_authority"] is False


def test_policy_loader_rejects_any_enabled_authority(tmp_path: Path) -> None:
    policy = _policy()
    policy["authority"] = {"source_delete_authority": True}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="authority"):
        load_lifecycle_policy(path)
