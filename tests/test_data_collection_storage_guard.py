import os
import time
from pathlib import Path

from scripts.ops import data_collection_storage_guard as src


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


def test_safe_space_recovery_deletes_only_bounded_safe_candidates(tmp_path: Path) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    root.mkdir(parents=True)
    registry = tmp_path / "master_bot_registry.json"
    registry.write_text('{"sub_bots":[]}', encoding="utf-8")

    duplicate = root / "shadow.local_fallback.jsonl"
    duplicate.write_bytes(b"duplicate")
    canonical = root / "shadow"
    canonical.write_bytes(b"canonical")
    stale_tmp = root / "nested" / "collector.partial"
    stale_tmp.parent.mkdir(parents=True)
    stale_tmp.write_bytes(b"partial")
    fresh_tmp = root / "fresh.partial"
    fresh_tmp.write_bytes(b"fresh")
    old_ts = time.time() - (8 * 3600)
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

    assert preview["safe_space_recovery"]["candidate_count"] == 2
    assert preview["safe_space_recovery"]["selected_count"] == 2
    assert preview["safe_space_recovery"]["by_reason"]["duplicate_local_fallback_artifact"]["count"] == 1
    assert preview["safe_space_recovery"]["by_reason"]["stale_partial_or_temp_artifact"]["count"] == 1

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

    assert applied["safe_space_recovery"]["deleted_count"] == 2
    assert not duplicate.exists()
    assert canonical.exists()
    assert not stale_tmp.exists()
    assert fresh_tmp.exists()
    assert src._is_protected_volume(Path("/Volumes/VIDEO/schwab_trading_bot")) is True


def test_safe_space_recovery_stops_when_target_free_space_is_met(tmp_path: Path) -> None:
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
