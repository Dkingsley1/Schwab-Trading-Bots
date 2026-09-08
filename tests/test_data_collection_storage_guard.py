import os
import time
from pathlib import Path

from scripts.ops import data_collection_storage_guard as src
import gzip
import json
import pytest


@pytest.fixture(autouse=True)
def idle_probe(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        src.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )


def test_quant_research_collectors_use_lighter_storage_profile() -> None:
    row = {
        "slot_kind": "graph_attention_cross_asset_spillover",
        "bot_role": "signal_sub_bot",
        "data_label_contract_version": "quant_research_labels_v1",
        "data_intake_collections": ["mlx_graph_library_profile"],
    }

    assert src._collector_kind(row) == "quant_research"

    profile = src._guard_profile("throttle", "quant_research")

    assert profile["capture_mode"] == "metadata_only"
    assert profile["sample_rate"] <= 0.08
    assert profile["max_daily_storage_mb"] <= 20


@pytest.mark.parametrize("compressed", [False, True])
def test_duplicate_removal_requires_full_content_and_durable_proof(
    tmp_path, compressed
):
    source = tmp_path / "events.jsonl.local_fallback"
    canonical = tmp_path / ("events.jsonl.gz" if compressed else "events.jsonl")
    data = b'{"event":1}\n' * 100
    source.write_bytes(data)
    canonical.write_bytes(gzip.compress(data) if compressed else data)
    proof = src._remove_verified_duplicate(source, canonical)
    assert proof["full_content_match"] and proof["source_removed"]
    assert not source.exists()
    assert canonical.exists()
    durable = json.loads(Path(proof["proof_path"]).read_text())
    assert durable["sha256"] == proof["sha256"]
    assert durable["source_removed"] is False
    assert not src._duplicate_fallback_files(tmp_path)


def test_divergent_or_open_duplicate_is_preserved(tmp_path, monkeypatch):
    source = tmp_path / "events.jsonl.local_fallback"
    canonical = tmp_path / "events.jsonl"
    source.write_bytes(b"same-prefix-A")
    canonical.write_bytes(b"same-prefix-B")
    with pytest.raises(ValueError, match="content_mismatch"):
        src._remove_verified_duplicate(source, canonical)
    assert source.read_bytes() == b"same-prefix-A"
    canonical.write_bytes(source.read_bytes())
    from types import SimpleNamespace

    monkeypatch.setattr(
        src.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="123", stderr=""),
    )
    with pytest.raises(ValueError, match="idle_probe"):
        src._remove_verified_duplicate(source, canonical)
    assert source.exists()


def test_duplicate_preserved_on_proof_failure_or_source_change(tmp_path, monkeypatch):
    source = tmp_path / "events.jsonl.local_fallback"
    canonical = tmp_path / "events.jsonl"
    source.write_bytes(b"original")
    canonical.write_bytes(b"original")
    real_write = src.write_payload
    monkeypatch.setattr(src, "write_payload", lambda *a, **k: None)
    with pytest.raises(ValueError, match="proof_publication"):
        src._remove_verified_duplicate(source, canonical)
    assert source.exists()

    def change_during_proof(path, payload):
        real_write(path, payload)
        source.write_bytes(b"changed!")

    monkeypatch.setattr(src, "write_payload", change_during_proof)
    with pytest.raises(ValueError, match="changed_before_release"):
        src._remove_verified_duplicate(source, canonical)
    assert source.read_bytes() == b"changed!"


def test_archived_fallbacks_remain_inventory_not_active_cleanup(tmp_path):
    root = tmp_path / "archive_root"
    cold = root / "cold_archive/storage_split_brain/old.jsonl.local_fallback"
    cold.parent.mkdir(parents=True)
    cold.write_bytes(b"irreplaceable history")
    registry = tmp_path / "registry.json"
    registry.write_text('{"sub_bots":[]}')
    payload = src.build_payload(
        external_root=root,
        registry_path=registry,
        warn_gb=120,
        throttle_gb=80,
        critical_gb=40,
        apply=False,
        cleanup_duplicates=True,
        space_recovery=True,
    )
    duplicate = payload["duplicate_cleanup"]
    assert duplicate["candidate_count"] == 0
    assert duplicate["archived_inventory"]["count"] == 1
    assert not duplicate["archived_inventory"]["deletion_allowed"]
    assert not duplicate["archived_inventory"]["reconciliation_verified"]
    assert cold.read_bytes() == b"irreplaceable history"


def test_duplicate_cleanup_protected_alias_never_reaches_disk_probe(
    tmp_path, monkeypatch
):
    alias = tmp_path / "reserved"
    alias.symlink_to("/Volumes/VIDEO")
    monkeypatch.setattr(
        src, "_disk_usage", lambda path: pytest.fail("protected metadata probe")
    )
    payload = src.build_payload(
        external_root=alias,
        registry_path=tmp_path / "registry.json",
        warn_gb=120,
        throttle_gb=80,
        critical_gb=40,
        apply=True,
        cleanup_duplicates=True,
    )
    assert payload["overall_status"] == "blocked"


def test_safe_space_recovery_deletes_only_bounded_safe_candidates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    root.mkdir(parents=True)
    registry = tmp_path / "master_bot_registry.json"
    registry.write_text('{"sub_bots":[]}', encoding="utf-8")

    duplicate = root / "shadow.local_fallback.jsonl"
    duplicate.write_bytes(b"duplicate")
    canonical = root / "shadow"
    canonical.write_bytes(b"duplicate")
    stale_tmp = root / "nested" / "collector.partial"
    stale_tmp.parent.mkdir(parents=True)
    stale_tmp.write_bytes(b"partial")
    fresh_tmp = root / "fresh.partial"
    fresh_tmp.write_bytes(b"fresh")
    old_ts = time.time() - (8 * 3600)
    os.utime(duplicate, (old_ts, old_ts))
    os.utime(stale_tmp, (old_ts, old_ts))

    preview = src.build_payload(
        external_root=root,
        registry_path=registry,
        warn_gb=120.0,
        throttle_gb=80.0,
        critical_gb=40.0,
        apply=False,
        cleanup_duplicates=True,
        space_recovery=True,
        space_recovery_max_delete_gb=1.0,
        space_recovery_target_free_gb=10000.0,
        space_recovery_min_age_hours=6.0,
    )

    assert preview["safe_space_recovery"]["candidate_count"] == 1
    assert preview["safe_space_recovery"]["selected_count"] == 1
    assert (
        preview["safe_space_recovery"]["by_reason"][
            "duplicate_local_fallback_artifact"
        ]["count"]
        == 1
    )
    assert (
        "stale_partial_or_temp_artifact"
        not in preview["safe_space_recovery"]["by_reason"]
    )

    applied = src.build_payload(
        external_root=root,
        registry_path=registry,
        warn_gb=120.0,
        throttle_gb=80.0,
        critical_gb=40.0,
        apply=True,
        cleanup_duplicates=True,
        space_recovery=True,
        space_recovery_max_delete_gb=1.0,
        space_recovery_target_free_gb=10000.0,
        space_recovery_min_age_hours=6.0,
    )

    assert applied["safe_space_recovery"]["deleted_count"] == 1
    assert not duplicate.exists()
    assert canonical.exists()
    assert stale_tmp.exists()
    assert fresh_tmp.exists()
    assert src._is_protected_volume(Path("/Volumes/VIDEO/schwab_trading_bot")) is True


def test_safe_space_recovery_stops_when_target_free_space_is_met(
    tmp_path: Path,
) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    root.mkdir(parents=True)
    registry = tmp_path / "master_bot_registry.json"
    registry.write_text('{"sub_bots":[]}', encoding="utf-8")
    canonical = root / "events.jsonl"
    fallback = root / "events.jsonl.local_fallback"
    canonical.write_bytes(b"canonical")
    fallback.write_bytes(b"duplicate")

    payload = src.build_payload(
        external_root=root,
        registry_path=registry,
        warn_gb=120.0,
        throttle_gb=80.0,
        critical_gb=40.0,
        apply=False,
        cleanup_duplicates=True,
        space_recovery=True,
        space_recovery_max_delete_gb=1.0,
        space_recovery_target_free_gb=0.001,
        space_recovery_min_age_hours=0.0,
    )

    assert payload["safe_space_recovery"]["candidate_count"] == 1
    assert payload["safe_space_recovery"]["target_free_deficit_gb"] == 0.0
    assert payload["safe_space_recovery"]["effective_max_delete_gb"] == 0.0
    assert payload["safe_space_recovery"]["selected_count"] == 0
    assert fallback.exists()


def test_safe_space_recovery_selects_single_oversized_stale_temp_below_target() -> None:
    stale_tmp = {
        "path": "/Volumes/BOT_LOGS/schwab_trading_bot/data/.jsonl_link.sqlite3.tmp",
        "relative_path": "data/.jsonl_link.sqlite3.tmp",
        "reason": "stale_partial_or_temp_artifact",
        "size_bytes": int(40 * 1024**3),
        "size_gb": 40.0,
        "age_hours": 829.0,
    }

    selected = src._select_space_recovery_candidates(
        [stale_tmp],
        max_delete_gb=29.0,
        jumbo_duplicate_gb=12.0,
        stale_temp_overshoot_gb=48.0,
    )

    assert selected == [stale_tmp]
    assert selected[0]["selected_over_wave_cap"] is True
    assert (
        selected[0]["selection_reason"]
        == "single_stale_partial_or_temp_artifact_to_restore_reserve"
    )


def test_safe_space_recovery_preserves_unverified_stateful_history_under_pressure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    data_root = root / "data"
    data_root.mkdir(parents=True)
    registry = tmp_path / "master_bot_registry.json"
    registry.write_text('{"sub_bots":[]}', encoding="utf-8")

    active = data_root / "bot_channel_queue.sqlite3"
    old_corrupt = data_root / "bot_channel_queue.sqlite3.corrupt-20260730150925006959"
    old_backup = (
        data_root / "jsonl_link.sqlite3.pre_local_failover_20260731T230144310331Z.bak"
    )
    fresh_corrupt = data_root / "snapshot_context.sqlite3.corrupt_20260907120000000000"
    active.write_bytes(b"active")
    old_corrupt.write_bytes(b"corrupt" * 300_000)
    old_backup.write_bytes(b"backup" * 128)
    fresh_corrupt.write_bytes(b"fresh")
    old_ts = time.time() - (48 * 3600)
    os.utime(old_corrupt, (old_ts, old_ts))
    os.utime(old_backup, (old_ts, old_ts))

    preview = src.build_payload(
        external_root=root,
        registry_path=registry,
        warn_gb=120.0,
        throttle_gb=80.0,
        critical_gb=40.0,
        apply=False,
        cleanup_duplicates=True,
        space_recovery=True,
        space_recovery_max_delete_gb=0.001,
        space_recovery_target_free_gb=10000.0,
        space_recovery_min_age_hours=6.0,
        space_recovery_jumbo_stateful_debris_gb=1.0,
    )

    by_reason = preview["safe_space_recovery"]["by_reason"]
    assert "old_stateful_corrupt_sqlite_artifact" not in by_reason
    assert "old_stateful_failover_backup_artifact" not in by_reason
    assert preview["safe_space_recovery"]["selected_count"] == 0

    applied = src.build_payload(
        external_root=root,
        registry_path=registry,
        warn_gb=120.0,
        throttle_gb=80.0,
        critical_gb=40.0,
        apply=True,
        cleanup_duplicates=True,
        space_recovery=True,
        space_recovery_max_delete_gb=0.001,
        space_recovery_target_free_gb=10000.0,
        space_recovery_min_age_hours=6.0,
        space_recovery_jumbo_stateful_debris_gb=1.0,
    )

    assert applied["safe_space_recovery"]["deleted_count"] == 0
    assert active.exists()
    assert old_corrupt.exists()
    assert old_backup.exists()
    assert fresh_corrupt.exists()
