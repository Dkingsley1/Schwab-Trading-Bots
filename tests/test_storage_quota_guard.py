from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.ops import storage_quota_guard as src


def test_stateful_sql_breakdown_counts_primary_queue_and_shards_once_through_symlinks(
    tmp_path: Path,
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    shard_root = physical / "shards"
    shard_root.mkdir()
    (shard_root / "trading.sqlite3").write_bytes(b"s" * 11)
    primary = physical / "jsonl_link.sqlite3"
    primary.write_bytes(b"p" * 13)
    queue = physical / "bot_channel_queue.sqlite3"
    queue.write_bytes(b"q" * 17)

    data = tmp_path / "data"
    data.mkdir()
    (data / "sql_link_shards").symlink_to(shard_root, target_is_directory=True)
    (data / "jsonl_link.sqlite3").symlink_to(primary)
    (data / "bot_channel_queue.sqlite3").symlink_to(queue)
    local_data = tmp_path / "local_fallback_storage" / "data"
    local_data.mkdir(parents=True)
    (local_data / "jsonl_link.sqlite3").symlink_to(primary)
    (local_data / "bot_channel_queue.sqlite3").symlink_to(queue)

    payload = src._stateful_sql_shard_breakdown(tmp_path)

    assert payload["shard_bytes"] == 11
    assert payload["primary_cache_bytes"] == 13
    assert payload["queue_bytes"] == 17
    assert payload["total_bytes"] == 41
    assert {row["component"] for row in payload["stateful_components"]} == {
        "primary_compatibility_cache",
        "queue",
    }


GIB = 1024**3


def test_shard_accounting_includes_fallback_with_empty_canonical_root(tmp_path):
    canonical = tmp_path / "data/sql_link_shards"
    fallback = tmp_path / "local_fallback_storage/data/sql_link_shards"
    canonical.mkdir(parents=True)
    fallback.mkdir(parents=True)
    (fallback / "trading.sqlite3").write_bytes(b"x" * 19)
    payload = src._stateful_sql_shard_breakdown(tmp_path)
    assert payload["shard_bytes"] == 19
    assert payload["total_bytes"] == 19
    assert payload["observed_shard_roots"] == [str(canonical), str(fallback)]


def test_distinct_resident_copies_count_but_aliases_and_hardlinks_do_not(tmp_path):
    canonical = tmp_path / "data/sql_link_shards"
    fallback = tmp_path / "local_fallback_storage/data/sql_link_shards"
    canonical.mkdir(parents=True)
    fallback.mkdir(parents=True)
    original = canonical / "trading.sqlite3"
    original.write_bytes(b"x" * 19)
    (fallback / "copy.sqlite3").write_bytes(original.read_bytes())
    (fallback / "alias.sqlite3").symlink_to(original)
    os.link(original, fallback / "hardlink.sqlite3")
    payload = src._stateful_sql_shard_breakdown(tmp_path)
    assert payload["shard_bytes"] == 38
    assert payload["total_bytes"] == 38


def test_fallback_only_root_is_observed(tmp_path):
    fallback = tmp_path / "local_fallback_storage/data/sql_link_shards"
    fallback.mkdir(parents=True)
    (fallback / "api_ingress.sqlite3").write_bytes(b"x" * 23)
    payload = src._stateful_sql_shard_breakdown(tmp_path)
    assert payload["root_exists"] is True
    assert payload["support_bytes"] == 23
    assert payload["root"] == str(fallback)


def test_forbidden_shard_root_rejected_before_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(
        src, "inspect_storage_path", lambda path: {"status": "forbidden"}
    )

    def unexpected_resolve(*args, **kwargs):
        pytest.fail("forbidden route must not be resolved or inspected")

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)
    with pytest.raises(RuntimeError, match="protected_sql_shard_route"):
        src._stateful_sql_shard_breakdown(tmp_path)


def test_compressed_history_counts_routed_archives_once(tmp_path):
    root, cold = tmp_path / "decisions", tmp_path / "cold"
    root.mkdir()
    cold.mkdir()
    archive = cold / "history.jsonl.gz"
    archive.write_bytes(b"x" * 23)
    (root / "history.jsonl.gz").symlink_to(archive)
    (root / "alias.jsonl.gz").symlink_to(archive)
    (root / "local.jsonl.gz").write_bytes(b"y" * 17)
    assert src._compressed_archive_bytes(root) == 40
    assert archive.read_bytes() == b"x" * 23


def test_archive_alias_cannot_exempt_raw_or_missing_target(tmp_path):
    root = tmp_path / "decisions"
    root.mkdir()
    raw = tmp_path / "live.jsonl"
    raw.write_bytes(b"x" * 31)
    (root / "disguised.jsonl.gz").symlink_to(raw)
    (root / "broken.jsonl.gz").symlink_to(tmp_path / "absent.gz")
    assert src._compressed_archive_bytes(root) == 0


def test_protected_archive_target_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "decisions"
    root.mkdir()
    (root / "history.jsonl.gz").write_bytes(b"x" * 23)
    original = src.inspect_storage_path
    monkeypatch.setattr(
        src,
        "inspect_storage_path",
        lambda path: (
            {"status": "forbidden"} if path.suffix == ".gz" else original(path)
        ),
    )
    with pytest.raises(RuntimeError, match="protected_archive_target"):
        src._compressed_archive_bytes(root)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_health(
    project_root: Path,
    *,
    decisions_gb: float = 0.0,
    governance_gb: float = 30.0,
    explanations_gb: float = 0.0,
    sql_link_shards_gb: float = 0.0,
    hot_path_over_budget_bytes: int = 0,
    hot_lane_mode: str = "full_decision_evidence",
    hot_lane_status: str = "ready",
) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "storage_tier_policy_latest.json",
        {
            "overall_status": "ready",
            "pressure": {
                "hot_path_over_budget_bytes": hot_path_over_budget_bytes,
                "live_hot_path_bytes": 4 * GIB,
                "hot_budget_bytes": 25 * GIB,
            },
            "by_family": {
                "decisions": {"bytes": int(decisions_gb * GIB)},
                "decision_explanations": {"bytes": int(explanations_gb * GIB)},
                "sql_link_shards": {"bytes": int(sql_link_shards_gb * GIB)},
            },
            "by_service_role": {
                "governance_telemetry": {"bytes": int(governance_gb * GIB)},
            },
        },
    )
    _write_json(
        health / "hot_lane_retention_control_latest.json",
        {
            "ok": True,
            "overall_status": hot_lane_status,
            "mode": hot_lane_mode,
        },
    )
    _write_json(health / "data_collection_storage_guard_latest.json", {})


def test_quota_guard_excludes_bounded_current_day_governance_when_full_evidence_hot_path_is_green(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=30.0)
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src,
        "_active_current_day_governance_channel_bytes",
        lambda _project_root: 24 * GIB,
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )

    payload = src.build_payload(tmp_path)

    governance = next(
        row for row in payload["lanes"] if row["family"] == "governance_telemetry"
    )
    assert payload["overall_status"] == "ready"
    assert governance["status"] == "ready"
    assert governance["used_gb"] == 6.0
    assert governance["adjustments"][0]["reason"] == (
        "exclude_bounded_current_day_active_governance_channels_under_green_full_evidence_hot_lane"
    )
    assert (
        payload["active_hot_buffer_containment"][
            "hot_lane_full_evidence_current_day_governance_relief"
        ]
        is True
    )


def test_quota_guard_extends_current_day_governance_buffer_for_green_full_evidence_soak(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=33.5)
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src,
        "_active_current_day_governance_channel_bytes",
        lambda _project_root: int(33.5 * GIB),
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )

    payload = src.build_payload(tmp_path)

    governance = next(
        row for row in payload["lanes"] if row["family"] == "governance_telemetry"
    )
    assert payload["overall_status"] == "ready"
    assert payload["quota_summary"]["soft_breaches"] == 0
    assert governance["status"] == "ready"
    assert governance["used_gb"] == 0.0
    assert governance["adjustments"][0]["reason"] == (
        "exclude_extended_current_day_active_governance_channels_under_green_full_evidence_hot_lane"
    )
    assert governance["adjustments"][0]["full_evidence_max_gb"] == 48.0
    assert governance["adjustments"][0]["legacy_governance_after_current_day_gb"] == 0.0


def test_quota_guard_does_not_extend_current_day_governance_buffer_over_full_evidence_cap(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=54.0)
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src,
        "_active_current_day_governance_channel_bytes",
        lambda _project_root: int(54.0 * GIB),
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )

    payload = src.build_payload(tmp_path)

    governance = next(
        row for row in payload["lanes"] if row["family"] == "governance_telemetry"
    )
    assert payload["overall_status"] == "blocked"
    assert governance["status"] == "blocked"
    assert governance["used_gb"] == 30.0
    assert governance["adjustments"][0]["reason"] == (
        "exclude_bounded_current_day_active_governance_channels_under_green_full_evidence_hot_lane"
    )
    assert governance["adjustments"][0]["gb"] == 24.0


def test_quota_guard_does_not_extend_current_day_governance_buffer_when_old_telemetry_exceeds_soft(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=42.0)
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src,
        "_active_current_day_governance_channel_bytes",
        lambda _project_root: int(33.5 * GIB),
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )

    payload = src.build_payload(tmp_path)

    governance = next(
        row for row in payload["lanes"] if row["family"] == "governance_telemetry"
    )
    assert payload["overall_status"] == "blocked"
    assert governance["status"] == "blocked"
    assert governance["used_gb"] == 18.0
    assert governance["adjustments"][0]["reason"] == (
        "exclude_bounded_current_day_active_governance_channels_under_green_full_evidence_hot_lane"
    )
    assert governance["adjustments"][0]["legacy_governance_after_current_day_gb"] == 8.5


def test_quota_guard_keeps_governance_blocked_when_hot_path_is_over_budget(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=30.0, hot_path_over_budget_bytes=1)
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src,
        "_active_current_day_governance_channel_bytes",
        lambda _project_root: 24 * GIB,
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )

    payload = src.build_payload(tmp_path)

    governance = next(
        row for row in payload["lanes"] if row["family"] == "governance_telemetry"
    )
    assert payload["overall_status"] == "blocked"
    assert governance["status"] == "blocked"
    assert governance["used_gb"] == 30.0
    assert governance["adjustments"] == []
    assert (
        payload["active_hot_buffer_containment"][
            "hot_lane_full_evidence_current_day_governance_relief"
        ]
        is False
    )


def test_quota_guard_excludes_current_day_decision_buffer_under_green_hot_lane_watch(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(
        tmp_path,
        decisions_gb=45.0,
        governance_gb=0.0,
        hot_lane_mode="watch",
        hot_lane_status="watching",
    )
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 24 * GIB
    )
    monkeypatch.setattr(
        src, "_active_current_day_governance_channel_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )

    payload = src.build_payload(tmp_path)

    decisions = next(row for row in payload["lanes"] if row["family"] == "decisions")
    assert payload["overall_status"] == "ready"
    assert payload["quota_summary"]["hard_breaches"] == 0
    assert decisions["status"] == "advisory"
    assert decisions["used_gb"] == 21.0
    assert decisions["adjustments"][0]["reason"] == (
        "exclude_current_day_active_decision_buffer_under_green_hot_lane_watch"
    )
    assert (
        payload["active_hot_buffer_containment"][
            "hot_lane_watch_current_day_decision_relief"
        ]
        is True
    )


def test_quota_guard_excludes_current_day_explanations_when_hot_lane_retention_active(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(
        tmp_path,
        governance_gb=0.0,
        explanations_gb=30.0,
        hot_lane_mode="emergency_hot_thin",
        hot_lane_status="critical",
    )
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_governance_channel_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 20 * GIB
    )

    payload = src.build_payload(tmp_path)

    explanations = next(
        row for row in payload["lanes"] if row["family"] == "decision_explanations"
    )
    assert payload["overall_status"] == "ready"
    assert explanations["status"] == "ready"
    assert explanations["used_gb"] == 14.0
    assert explanations["adjustments"][0]["reason"] == (
        "exclude_bounded_current_day_explanation_buffer_under_hot_lane_retention"
    )
    assert explanations["adjustments"][0]["hard_quota_protected"] is True
    assert (
        payload["active_hot_buffer_containment"]["active_current_day_explanation_gb"]
        == 20.0
    )
    assert (
        payload["active_hot_buffer_containment"][
            "active_explanation_buffer_allowance_gb"
        ]
        == 16.0
    )


def test_quota_guard_does_not_hide_hard_explanation_breach(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(
        tmp_path,
        governance_gb=0.0,
        explanations_gb=60.0,
        hot_lane_mode="emergency_hot_thin",
        hot_lane_status="critical",
    )
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_governance_channel_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 40 * GIB
    )

    payload = src.build_payload(tmp_path)

    explanations = next(
        row for row in payload["lanes"] if row["family"] == "decision_explanations"
    )
    assert payload["overall_status"] == "blocked"
    assert explanations["status"] == "blocked"
    assert explanations["used_gb"] == 60.0
    assert explanations["adjustments"] == []


def test_quota_guard_manages_support_sql_soft_quota_when_core_is_below_quota(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=0.0, sql_link_shards_gb=377.0)
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_governance_channel_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src,
        "_stateful_sql_shard_breakdown",
        lambda _project_root: {
            "root": str(tmp_path / "data" / "sql_link_shards"),
            "root_exists": True,
            "root_free_gb": 128.0,
            "support_bytes": int(259 * GIB),
            "core_bytes": int(118 * GIB),
            "total_bytes": int(377 * GIB),
            "support_gb": 259.0,
            "core_gb": 118.0,
            "top_support_shards": [
                {"name": "jsonl_link_risk_support.sqlite3", "size_gb": 259.0}
            ],
            "top_core_shards": [
                {"name": "jsonl_link_crypto_trading.sqlite3", "size_gb": 83.0}
            ],
            "support_markers": ["risk_support"],
        },
    )

    payload = src.build_payload(tmp_path)

    sql_lane = next(
        row for row in payload["lanes"] if row["family"] == "sql_link_shards"
    )
    assert payload["overall_status"] == "ready"
    assert payload["quota_summary"]["advisory_breaches"] == 1
    assert sql_lane["status"] == "advisory"
    assert sql_lane["raw_used_gb"] == 377.0
    assert sql_lane["used_gb"] == 118.0
    assert sql_lane["managed_support_sql_relief"]["active"] is True
    assert (
        sql_lane["adjustments"][0]["reason"]
        == "exclude_managed_support_sql_shards_from_core_stateful_quota"
    )


def test_quota_guard_manages_support_sql_slight_overhard_when_core_and_free_space_are_safe(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=0.0, sql_link_shards_gb=384.0)
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_governance_channel_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src,
        "_stateful_sql_shard_breakdown",
        lambda _project_root: {
            "root": str(tmp_path / "data" / "sql_link_shards"),
            "root_exists": True,
            "root_free_gb": 128.0,
            "support_bytes": int(259 * GIB),
            "core_bytes": int(125 * GIB),
            "total_bytes": int(384 * GIB),
            "support_gb": 259.0,
            "core_gb": 125.0,
            "top_support_shards": [
                {"name": "jsonl_link_risk_support.sqlite3", "size_gb": 259.0}
            ],
            "top_core_shards": [
                {"name": "jsonl_link_crypto_trading.sqlite3", "size_gb": 83.0}
            ],
            "support_markers": ["risk_support"],
        },
    )

    payload = src.build_payload(tmp_path)

    sql_lane = next(
        row for row in payload["lanes"] if row["family"] == "sql_link_shards"
    )
    assert payload["overall_status"] == "ready"
    assert payload["quota_summary"]["advisory_breaches"] == 1
    assert sql_lane["status"] == "advisory"
    assert sql_lane["raw_used_gb"] == 384.0
    assert sql_lane["used_gb"] == 125.0
    assert sql_lane["managed_support_sql_relief"]["active"] is True
    assert sql_lane["managed_support_sql_relief"]["raw_hard_ratio"] == 1.011
    assert sql_lane["managed_support_sql_relief"]["blockers"] == []


def test_quota_guard_prefers_verified_sql_shard_usage_when_storage_tier_is_stale(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=0.0, sql_link_shards_gb=392.0)
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_governance_channel_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src,
        "_stateful_sql_shard_breakdown",
        lambda _project_root: {
            "root": str(tmp_path / "data" / "sql_link_shards"),
            "root_exists": True,
            "root_free_gb": 568.0,
            "support_bytes": int(26 * GIB),
            "core_bytes": int(13 * GIB),
            "total_bytes": int(39 * GIB),
            "support_gb": 26.0,
            "core_gb": 13.0,
            "top_support_shards": [
                {"name": "jsonl_link_risk_support.sqlite3", "size_gb": 26.0}
            ],
            "top_core_shards": [{"name": "jsonl_link_trading.sqlite3", "size_gb": 5.0}],
            "support_markers": ["risk_support"],
        },
    )

    payload = src.build_payload(tmp_path)

    sql_lane = next(
        row for row in payload["lanes"] if row["family"] == "sql_link_shards"
    )
    assert payload["overall_status"] == "ready"
    assert sql_lane["status"] == "ready"
    assert sql_lane["raw_used_gb"] == 39.0
    assert sql_lane["used_gb"] == 13.0
    assert (
        sql_lane["accounting_reconciliations"][0]["storage_tier_reported_gb"] == 392.0
    )
    assert sql_lane["accounting_reconciliations"][0]["verified_filesystem_gb"] == 39.0
    assert (
        sql_lane["adjustments"][0]["reason"]
        == "exclude_managed_support_sql_shards_from_core_stateful_quota"
    )


def test_quota_guard_does_not_manage_support_sql_far_overhard(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=0.0, sql_link_shards_gb=460.0)
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_governance_channel_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src,
        "_stateful_sql_shard_breakdown",
        lambda _project_root: {
            "root": str(tmp_path / "data" / "sql_link_shards"),
            "root_exists": True,
            "root_free_gb": 128.0,
            "support_bytes": int(320 * GIB),
            "core_bytes": int(140 * GIB),
            "total_bytes": int(460 * GIB),
            "support_gb": 320.0,
            "core_gb": 140.0,
            "top_support_shards": [
                {"name": "jsonl_link_risk_support.sqlite3", "size_gb": 320.0}
            ],
            "top_core_shards": [
                {"name": "jsonl_link_crypto_trading.sqlite3", "size_gb": 83.0}
            ],
            "support_markers": ["risk_support"],
        },
    )

    payload = src.build_payload(tmp_path)

    sql_lane = next(
        row for row in payload["lanes"] if row["family"] == "sql_link_shards"
    )
    assert payload["overall_status"] == "blocked"
    assert sql_lane["status"] == "blocked"
    assert sql_lane["managed_support_sql_relief"]["active"] is False
    assert (
        "raw_stateful_sql_above_managed_support_relief_ceiling"
        in sql_lane["managed_support_sql_relief"]["blockers"]
    )


def test_quota_guard_does_not_manage_support_sql_when_free_space_is_low(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=0.0, sql_link_shards_gb=377.0)
    monkeypatch.setattr(
        src, "_active_current_day_decision_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_governance_channel_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src, "_active_current_day_explanation_bytes", lambda _project_root: 0
    )
    monkeypatch.setattr(
        src,
        "_stateful_sql_shard_breakdown",
        lambda _project_root: {
            "root": str(tmp_path / "data" / "sql_link_shards"),
            "root_exists": True,
            "root_free_gb": 25.0,
            "support_bytes": int(259 * GIB),
            "core_bytes": int(118 * GIB),
            "total_bytes": int(377 * GIB),
            "support_gb": 259.0,
            "core_gb": 118.0,
            "top_support_shards": [
                {"name": "jsonl_link_risk_support.sqlite3", "size_gb": 259.0}
            ],
            "top_core_shards": [
                {"name": "jsonl_link_crypto_trading.sqlite3", "size_gb": 83.0}
            ],
            "support_markers": ["risk_support"],
        },
    )

    payload = src.build_payload(tmp_path)

    sql_lane = next(
        row for row in payload["lanes"] if row["family"] == "sql_link_shards"
    )
    assert payload["overall_status"] == "degraded"
    assert sql_lane["status"] == "degraded"
    assert sql_lane["used_gb"] == 377.0
    assert sql_lane["managed_support_sql_relief"]["active"] is False
    assert (
        "stateful_sql_root_free_below_support_relief_floor"
        in sql_lane["managed_support_sql_relief"]["blockers"]
    )
