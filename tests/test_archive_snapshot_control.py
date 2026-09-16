from core.archive_snapshot_control import (
    SnapshotCatalog,
    build_manifest,
    build_snapshot,
    plan_compaction,
)


def _entry(path: str, byte: str, size: int = 100) -> dict[str, object]:
    return {
        "path": path,
        "sha256": byte * 64,
        "size_bytes": size,
        "row_count": 10,
        "format": "jsonl",
    }


def test_snapshot_catalog_is_hash_bound_and_compare_and_swap_committed() -> None:
    manifest = build_manifest([_entry("b.jsonl", "b"), _entry("a.jsonl", "a")])
    snapshot = build_snapshot(
        manifest,
        schema_id="decision-v1",
        committed_at_utc="2026-08-21T12:00:00+00:00",
    )
    catalog = SnapshotCatalog()

    assert (
        catalog.commit(snapshot, expected_parent_snapshot_id=None)["committed"] is True
    )
    assert (
        catalog.commit(snapshot, expected_parent_snapshot_id=None)["committed"] is False
    )
    assert snapshot["source_delete_authority"] is False


def test_compaction_plan_retains_readable_manifest_and_never_deletes_sources() -> None:
    plan = plan_compaction(
        [_entry("a.jsonl", "a"), _entry("b.jsonl", "b")], target_bytes=500
    )

    assert plan["group_count"] == 1
    assert plan["human_readable_manifest_retained"] is True
    assert plan["source_delete_authority"] is False
