import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from scripts.ops import deep_cold_storage_layer as src


def test_deep_cold_move_to_second_cold_preserves_original_path_with_symlink(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external" / "schwab_trading_bot"
    second_cold = tmp_path / "VIDEO" / "schwab_trading_bot_cold"
    stale_file = project_root / "data" / "stale_stage" / "governance" / "execution_intents_20260728.jsonl.gz"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_bytes(b"retained-governance-evidence\n" * 64)

    monkeypatch.setattr(src, "resolve_external_storage", lambda: SimpleNamespace(external_root=external_root))

    payload = src.build_payload(
        project_root,
        apply=True,
        min_size_mb=0.000001,
        top_n=10,
        move_to_second_cold=True,
        second_cold_root=second_cold,
        max_move_gb=1.0,
        max_move_files=10,
    )

    move = payload["second_cold_move"]
    assert move["status"] == "ready"
    assert move["moved_files"] == 1
    assert stale_file.is_symlink()
    target = Path(move["actions"][0]["target"])
    assert target.exists()
    assert target.read_bytes() == b"retained-governance-evidence\n" * 64
    assert move["actions"][0]["verified_sha256_match"] is True
    assert stale_file.resolve(strict=True) == target.resolve(strict=True)
    assert payload["top_rows"][0]["source_replaced_with_symlink"] is True
    manifest = Path(payload["manifest_path"])
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_replaced_with_symlink"] is True


def test_deep_cold_move_resumes_verified_partial_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "cold" / "source.bin"
    payload = (b"resumable-deep-cold\n" * 1024) + b"tail"
    source.write_bytes(payload)
    target.parent.mkdir(parents=True)
    partial = target.with_name(f".{target.name}.tmp")
    partial.write_bytes(payload[: len(payload) // 2])

    result = src._copy_verify_then_symlink(source, target)

    assert result["source_replaced_with_symlink"] is True
    assert result["verified_sha256_match"] is True
    assert result["resumed_bytes"] == len(payload) // 2
    assert target.read_bytes() == payload
    assert source.resolve(strict=True) == target.resolve(strict=True)


def test_deep_cold_move_restarts_when_partial_prefix_does_not_match(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "cold" / "source.bin"
    payload = b"authoritative-source" * 1024
    source.write_bytes(payload)
    target.parent.mkdir(parents=True)
    partial = target.with_name(f".{target.name}.tmp")
    partial.write_bytes(b"wrong-prefix" * 128)

    result = src._copy_verify_then_symlink(source, target)

    assert result["source_replaced_with_symlink"] is True
    assert result["resumed_bytes"] == 0
    assert target.read_bytes() == payload


def test_video_cold_archive_subtree_is_allowed_only_when_explicitly_enabled(monkeypatch) -> None:
    root = Path("/Volumes/VIDEO/schwab_trading_bot_cold")

    monkeypatch.delenv("BOT_ALLOW_VIDEO_COLD_ARCHIVE", raising=False)
    assert src._is_protected_volume(Path("/Volumes/VIDEO")) is True
    assert src._is_protected_volume(root / "deep_cold" / "file.gz") is True

    monkeypatch.setenv("BOT_ALLOW_VIDEO_COLD_ARCHIVE", "1")
    monkeypatch.setenv("BOT_VIDEO_COLD_ARCHIVE_ROOT", str(root))
    assert src._is_protected_volume(Path("/Volumes/VIDEO")) is True
    assert src._is_protected_volume(root / "deep_cold" / "file.gz") is False


def test_active_capacity_source_prioritizes_external_hard_reserve_breach(monkeypatch, tmp_path: Path) -> None:
    gib = 1024**3
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    project_root.mkdir(parents=True)
    external_root.mkdir(parents=True)
    monkeypatch.setenv("BOT_DEEP_COLD_SOURCE_FREE_FLOOR_GB", "96")

    def fake_usage(path: Path) -> dict[str, int]:
        if path == external_root:
            return {"total_bytes": 900 * gib, "free_bytes": 40 * gib, "used_bytes": 860 * gib, "device_id": 2}
        return {"total_bytes": 900 * gib, "free_bytes": 140 * gib, "used_bytes": 760 * gib, "device_id": 1}

    monkeypatch.setattr(src, "_disk_usage_snapshot", fake_usage)

    selected, reason = src._active_capacity_source(project_root, external_root)

    assert selected == external_root
    assert reason == "external_filesystem_hard_reserve_breach"


def test_second_cold_cap_skips_oversized_file_and_moves_smaller_candidate(tmp_path: Path) -> None:
    large = tmp_path / "large.bin"
    small = tmp_path / "small.bin"
    large.write_bytes(b"l" * 2048)
    small.write_bytes(b"s" * 128)
    rows = [
        {"path": str(large), "relative_path": "project/large.bin", "size_bytes": large.stat().st_size},
        {"path": str(small), "relative_path": "project/small.bin", "size_bytes": small.stat().st_size},
    ]

    payload = src._apply_second_cold_moves(
        rows,
        second_cold_root=tmp_path / "cold",
        max_move_gb=512 / (1024**3),
        max_move_files=2,
        include_critical=True,
    )

    assert payload["moved_files"] == 1
    assert payload["skipped_over_cap_files"] == 1
    assert large.is_file()
    assert small.is_symlink()


def test_adaptive_release_selects_smallest_file_that_meets_need(tmp_path: Path) -> None:
    huge = tmp_path / "huge.bin"
    right_sized = tmp_path / "right-sized.bin"
    extra = tmp_path / "extra.bin"
    huge.write_bytes(b"h" * 4096)
    right_sized.write_bytes(b"r" * 1024)
    extra.write_bytes(b"e" * 256)
    rows = [
        {"path": str(huge), "relative_path": "project/huge.bin", "size_bytes": huge.stat().st_size},
        {"path": str(right_sized), "relative_path": "project/right.bin", "size_bytes": right_sized.stat().st_size},
        {"path": str(extra), "relative_path": "project/extra.bin", "size_bytes": extra.stat().st_size},
    ]

    payload = src._apply_second_cold_moves(
        rows,
        second_cold_root=tmp_path / "cold",
        max_move_gb=8192 / (1024**3),
        max_move_files=3,
        include_critical=True,
        release_target_gb=900 / (1024**3),
    )

    assert payload["release_target_met"] is True
    assert payload["moved_files"] == 1
    assert huge.is_file()
    assert right_sized.is_symlink()
    assert extra.is_file()


def test_local_quarantine_move_requires_runtime_maintenance_hold(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external" / "schwab_trading_bot"
    monkeypatch.setattr(src, "resolve_external_storage", lambda: SimpleNamespace(external_root=external_root))
    monkeypatch.setattr(src, "maintenance_hold_snapshot", lambda _root: {"active": False})

    payload = src.build_payload(
        project_root,
        apply=True,
        move_to_second_cold=True,
        second_cold_root=tmp_path / "cold",
        include_local_quarantine=True,
    )

    assert payload["ok"] is False
    assert payload["blocked_reason"] == "runtime_maintenance_hold_required_for_local_quarantine_move"


def test_adaptive_policy_uses_live_ratio_growth_and_destination_reserve(monkeypatch, tmp_path: Path) -> None:
    gib = 1024**3
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    def fake_usage(path: Path) -> dict[str, int | None]:
        if path == source:
            return {"total_bytes": 100 * gib, "used_bytes": 95 * gib, "free_bytes": 5 * gib, "device_id": 1}
        return {"total_bytes": 200 * gib, "used_bytes": 100 * gib, "free_bytes": 100 * gib, "device_id": 2}

    monkeypatch.setattr(src, "_disk_usage_snapshot", fake_usage)
    monkeypatch.setattr(
        src,
        "_fresh_growth_signal",
        lambda _root, now: {"effective_gb_per_day": 1.0, "source": "test"},
    )
    monkeypatch.setenv("BOT_DEEP_COLD_SOURCE_FREE_FLOOR_GB", "10")
    monkeypatch.setenv("BOT_DEEP_COLD_SOURCE_FREE_RATIO", "0.20")
    monkeypatch.setenv("BOT_DEEP_COLD_OPERATING_BUFFER_GB", "2")

    policy = src._adaptive_release_policy(
        project_root=tmp_path,
        source_path=source,
        destination_path=target,
        explicit_source_free_target_gb=0.0,
        explicit_release_target_gb=0.0,
        planning_horizon_days=10,
        max_move_gb=50.0,
        destination_reserve_gb=25.0,
        now=datetime.now(timezone.utc),
    )

    assert policy["status"] == "release_required"
    assert policy["target_free_gb"] == 20.0
    assert policy["requested_release_gb"] == 15.0
    assert policy["destination_headroom_gb"] == 75.0
    assert policy["effective_wave_cap_gb"] == 50.0
    assert policy["achievable_release_this_wave_gb"] == 15.0


def test_growth_signal_clamps_short_maintenance_window_spike(monkeypatch, tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    (health / "storage_growth_forecast_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "confidence": "sustained",
                "sustained_baseline": {"timestamp_utc": now.isoformat()},
                "elapsed_days": 0.025,
                "sustained_consumed_gb_per_day": 803.0,
                "consumed_gb_per_day": 803.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BOT_DEEP_COLD_GROWTH_FLOOR_GB_PER_DAY", "0.5")
    monkeypatch.setenv("BOT_DEEP_COLD_MIN_GROWTH_SLOPE_DAYS", "0.25")

    signal = src._fresh_growth_signal(tmp_path, now=now)

    assert signal["short_window_spike_clamped"] is True
    assert signal["effective_gb_per_day"] == 0.5
    assert signal["source"] == "fresh_short_window_growth_floor"


def test_adaptive_policy_refuses_same_filesystem_destination(monkeypatch, tmp_path: Path) -> None:
    gib = 1024**3
    monkeypatch.setattr(
        src,
        "_disk_usage_snapshot",
        lambda _path: {
            "total_bytes": 100 * gib,
            "used_bytes": 95 * gib,
            "free_bytes": 5 * gib,
            "device_id": 7,
        },
    )
    monkeypatch.setattr(
        src,
        "_fresh_growth_signal",
        lambda _root, now: {"effective_gb_per_day": 0.5, "source": "test"},
    )

    policy = src._adaptive_release_policy(
        project_root=tmp_path,
        source_path=tmp_path / "source",
        destination_path=tmp_path / "target",
        explicit_source_free_target_gb=25.0,
        explicit_release_target_gb=0.0,
        planning_horizon_days=30.0,
        max_move_gb=50.0,
        destination_reserve_gb=1.0,
        now=datetime.now(timezone.utc),
    )

    assert policy["status"] == "blocked_destination_same_filesystem"
    assert policy["same_filesystem"] is True
    assert policy["achievable_release_this_wave_gb"] == 0.0


def test_adaptive_policy_does_not_chase_soft_ratio_after_hard_reserve_is_met(monkeypatch, tmp_path: Path) -> None:
    gib = 1024**3
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    def fake_usage(path: Path) -> dict[str, int | None]:
        if path == source:
            return {"total_bytes": 100 * gib, "used_bytes": 85 * gib, "free_bytes": 15 * gib, "device_id": 1}
        return {"total_bytes": 200 * gib, "used_bytes": 100 * gib, "free_bytes": 100 * gib, "device_id": 2}

    monkeypatch.setattr(src, "_disk_usage_snapshot", fake_usage)
    monkeypatch.setattr(
        src,
        "_fresh_growth_signal",
        lambda _root, now: {"effective_gb_per_day": 1.0, "source": "test"},
    )
    monkeypatch.setenv("BOT_DEEP_COLD_SOURCE_FREE_FLOOR_GB", "10")
    monkeypatch.setenv("BOT_DEEP_COLD_SOURCE_FREE_RATIO", "0.20")
    monkeypatch.setenv("BOT_DEEP_COLD_OPERATING_BUFFER_GB", "2")

    policy = src._adaptive_release_policy(
        project_root=tmp_path,
        source_path=source,
        destination_path=target,
        explicit_source_free_target_gb=0.0,
        explicit_release_target_gb=0.0,
        planning_horizon_days=10,
        max_move_gb=50.0,
        destination_reserve_gb=25.0,
        now=datetime.now(timezone.utc),
    )

    assert policy["hard_target_free_gb"] == 12.0
    assert policy["target_free_gb"] == 20.0
    assert policy["hard_deficit_gb"] == 0.0
    assert policy["preferred_headroom_deficit_gb"] == 5.0
    assert policy["requested_release_gb"] == 0.0
    assert policy["status"] == "source_hard_reserve_satisfied_soft_headroom_watch"


def test_adaptive_move_uses_safe_rows_then_waits_for_hold(tmp_path: Path) -> None:
    safe = tmp_path / "safe.bak"
    guarded = tmp_path / "guarded.bin"
    safe.write_bytes(b"s" * 128)
    guarded.write_bytes(b"g" * 128)
    rows = [
        {
            "path": str(safe),
            "relative_path": "project/safe.bak",
            "size_bytes": 128,
            "source_device_id": 1,
            "economic_value": "high",
            "requires_maintenance_hold": False,
        },
        {
            "path": str(guarded),
            "relative_path": "project/guarded.bin",
            "size_bytes": 128,
            "source_device_id": 1,
            "economic_value": "low",
            "requires_maintenance_hold": True,
        },
    ]

    payload = src._apply_second_cold_moves(
        rows,
        second_cold_root=tmp_path / "cold",
        max_move_gb=1024 / (1024**3),
        max_move_files=2,
        include_critical=False,
        release_target_gb=200 / (1024**3),
        adaptive=True,
        source_device_id=1,
        maintenance_hold_active=False,
    )

    assert payload["status"] == "blocked"
    assert payload["reason"] == "adaptive_release_waiting_for_maintenance_hold"
    assert payload["moved_files"] == 1
    assert payload["maintenance_blocked_files"] == 1
    assert safe.is_symlink()
    assert guarded.is_file()


def test_adaptive_manifest_discovers_only_named_failover_backups(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external" / "schwab_trading_bot"
    failover_root = project_root / "local_fallback_storage" / "data"
    failover_root.mkdir(parents=True)
    backup = failover_root / "jsonl_link.sqlite3.pre_local_failover_20260731T230144Z.bak"
    active = failover_root / "jsonl_link.sqlite3"
    backup.write_bytes(b"b" * 128)
    active.write_bytes(b"a" * 128)
    monkeypatch.setattr(src, "resolve_external_storage", lambda: SimpleNamespace(external_root=external_root))

    payload = src.build_payload(
        project_root,
        adaptive=True,
        include_failover_backups=True,
        min_size_mb=0.000001,
        source_free_path=project_root,
        second_cold_root=tmp_path / "cold",
    )

    failover_rows = [
        row
        for row in payload["top_rows"]
        if row.get("artifact_class") == "superseded_verified_failover_backup"
    ]
    assert payload["summary"]["failover_backup_count"] == 1
    assert len(failover_rows) == 1
    assert Path(failover_rows[0]["path"]) == backup
    assert failover_rows[0]["requires_maintenance_hold"] is False
