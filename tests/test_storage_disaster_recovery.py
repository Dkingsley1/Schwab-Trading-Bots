import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import storage_disaster_recovery as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_probe_storage_skips_external_io_when_hot_route_is_pinned_local(monkeypatch) -> None:
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "0")
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/unresponsive")
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "/Volumes/unresponsive/schwab_trading_bot")
    monkeypatch.setattr(
        src,
        "resolve_external_storage",
        lambda: (_ for _ in ()).throw(AssertionError("external filesystem must not be probed")),
    )

    probe = src._probe_storage()

    assert probe["probe_skipped_external_io"] is True
    assert probe["hot_storage_available"] is True
    assert probe["external_required_for_hot_path"] is False


def test_storage_disaster_recovery_plans_mount_local_switch_and_snapshot(monkeypatch, tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external"})
    _write_json(health / "storage_mount_guard_latest.json", {"storage_mode": "external"})
    local_root = tmp_path / "local_fallback_storage"
    (local_root / "governance").mkdir(parents=True)
    (local_root / "logs").mkdir(parents=True)
    (local_root / "data").mkdir(parents=True)
    (local_root / "data" / "snapshot_context.sqlite3").write_text("db", encoding="utf-8")

    probe = {
        "external_available": False,
        "target_volume_present": True,
        "target_volume_mounted": False,
        "target_volume_device_identifier": "disk9s1",
        "external_unavailable_reason": "volume_unmounted",
    }
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(local_root))
    monkeypatch.setattr(src, "_probe_storage", lambda: dict(probe))

    payload, _state = src.build_payload(
        tmp_path,
        apply=False,
        recovery_root=tmp_path / "recovery",
        state_path=tmp_path / "state.json",
        mount_cooldown_seconds=120.0,
        snapshot_cooldown_seconds=3600.0,
    )

    assert payload["overall_status"] == "blocked"
    assert payload["mount_attempt"]["skipped_reason"] == "apply_disabled"
    assert payload["switch_local"]["target_mode"] == "local"
    assert payload["recovery_snapshot"]["selected_paths"]
    assert any("local fallback" in action for action in payload["recommended_actions"])


def test_storage_disaster_recovery_applies_restore_when_external_returns(monkeypatch, tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "local_fallback"})
    _write_json(health / "storage_mount_guard_latest.json", {"storage_mode": "local_fallback"})

    probes = iter(
        [
            {
                "external_available": True,
                "target_volume_present": True,
                "target_volume_mounted": True,
                "target_volume_device_identifier": "disk9s1",
                "external_unavailable_reason": "ok",
            },
            {
                "external_available": True,
                "target_volume_present": True,
                "target_volume_mounted": True,
                "target_volume_device_identifier": "disk9s1",
                "external_unavailable_reason": "ok",
            },
            {
                "external_available": True,
                "target_volume_present": True,
                "target_volume_mounted": True,
                "target_volume_device_identifier": "disk9s1",
                "external_unavailable_reason": "ok",
            },
            {
                "external_available": True,
                "target_volume_present": True,
                "target_volume_mounted": True,
                "target_volume_device_identifier": "disk9s1",
                "external_unavailable_reason": "ok",
            },
        ]
    )
    monkeypatch.setattr(src, "_probe_storage", lambda: dict(next(probes)))
    monkeypatch.setenv("BOT_LOGS_RECOVERY_AUTO_FAILBACK_EXTERNAL", "1")

    calls: list[str] = []

    def _fake_switch(project_root: Path, target_mode: str, *, apply: bool) -> dict:
        calls.append(target_mode)
        _write_json(health / "storage_failback_sync_latest.json", {"mode": "external"})
        return {"attempted": True, "ok": True, "target_mode": target_mode}

    monkeypatch.setattr(src, "_switch_storage_mode", _fake_switch)
    monkeypatch.setattr(src, "_transactional_curated_restore", lambda source_root, external_root, *, apply, project_root=src.PROJECT_ROOT: {"attempted": True, "ok": True, "source_root": str(source_root), "target_root": str(external_root)})
    monkeypatch.setattr(src, "_sync_storage_target_override", lambda project_root, probe, *, apply: {"attempted": True, "ok": True, "changed": True})
    monkeypatch.setattr(src, "_sync_finder_shortcuts", lambda project_root, *, apply: {"attempted": True, "ok": True})

    payload, _state = src.build_payload(
        tmp_path,
        apply=True,
        recovery_root=tmp_path / "recovery",
        state_path=tmp_path / "state.json",
        mount_cooldown_seconds=120.0,
        snapshot_cooldown_seconds=3600.0,
    )

    assert calls == ["external"]
    assert payload["restore_external"]["ok"] is True
    assert payload["current_storage_mode"] == "external"
    assert payload["overall_status"] == "ready"


def test_storage_disaster_recovery_preserves_pinned_local_route(monkeypatch, tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "local_fallback"})
    override = tmp_path / "config" / ".env.storage_override"
    override.parent.mkdir(parents=True)
    override.write_text("BOT_LOGS_PREFER_EXTERNAL=0\n", encoding="utf-8")
    probe = {
        "external_available": True,
        "target_volume_present": True,
        "target_volume_mounted": True,
        "external_root": str(tmp_path / "external"),
        "external_unavailable_reason": "ok",
    }
    monkeypatch.setattr(src, "_probe_storage", lambda: dict(probe))
    monkeypatch.setattr(
        src,
        "_transactional_curated_restore",
        lambda source_root, external_root, *, apply, project_root=src.PROJECT_ROOT: {
            "attempted": False,
            "ok": False,
            "skipped_reason": "writer_not_quiet",
        },
    )
    monkeypatch.setattr(
        src,
        "_switch_storage_mode",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local pin must suppress external failback")),
    )
    monkeypatch.setattr(
        src,
        "_sync_storage_target_override",
        lambda project_root, probe, *, apply: {"attempted": False, "ok": True, "changed": False},
    )

    payload, _state = src.build_payload(
        tmp_path,
        apply=True,
        recovery_root=tmp_path / "recovery",
        state_path=tmp_path / "state.json",
        mount_cooldown_seconds=120.0,
        snapshot_cooldown_seconds=3600.0,
    )

    assert payload["overall_status"] == "ready"
    assert payload["current_storage_mode"] == "local_fallback"
    assert payload["route_policy"]["local_route_pinned"] is True
    assert payload["restore_external"]["skipped_reason"] == "local_route_pinned"


def test_current_storage_mode_prefers_physical_local_routes_over_stale_external_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external"})
    local_root = tmp_path / "local_fallback_storage"
    external_root = tmp_path / "external"
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(local_root))
    for relative_path in src.TRACKED_SQLITE_ROUTES:
        target = local_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        route = tmp_path / relative_path
        route.parent.mkdir(parents=True, exist_ok=True)
        route.symlink_to(target)

    mode = src._current_storage_mode(
        tmp_path,
        probe={"external_available": True, "external_root": str(external_root)},
    )

    assert mode == "local_fallback"


def test_take_curated_snapshot_copies_only_important_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_LOGS_RECOVERY_MIN_FREE_AFTER_SNAPSHOT_GB", "0")
    local_root = tmp_path / "local_fallback_storage"
    (local_root / "governance").mkdir(parents=True)
    (local_root / "governance" / "one.json").write_text("{}", encoding="utf-8")
    (local_root / "logs").mkdir(parents=True)
    (local_root / "logs" / "main.log").write_text("log", encoding="utf-8")
    (local_root / "data").mkdir(parents=True)
    (local_root / "data" / "snapshot_context.sqlite3").write_text("db", encoding="utf-8")
    (local_root / "data" / "giant_raw.sqlite3").write_text("ignore", encoding="utf-8")

    payload = src._take_curated_snapshot(
        local_root,
        tmp_path / "recovery",
        apply=True,
        state={},
        cooldown_seconds=0.0,
        project_root=tmp_path,
    )

    latest = tmp_path / "recovery" / "latest"
    assert payload["ok"] is True
    assert (latest / "governance" / "one.json").exists()
    assert (latest / "logs" / "main.log").exists()
    assert (latest / "data" / "snapshot_context.sqlite3").exists()
    assert not (latest / "data" / "giant_raw.sqlite3").exists()


def test_take_curated_snapshot_excludes_symlinked_external_routes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_LOGS_RECOVERY_MIN_FREE_AFTER_SNAPSHOT_GB", "0")
    local_root = tmp_path / "local_fallback_storage"
    external_root = tmp_path / "external"
    (external_root / "governance").mkdir(parents=True)
    (external_root / "governance" / "external.jsonl").write_text("external", encoding="utf-8")
    local_root.mkdir()
    (local_root / "governance").symlink_to(external_root / "governance")
    (local_root / "logs").mkdir()
    (local_root / "logs" / "local.log").write_text("local", encoding="utf-8")
    (local_root / "logs" / "external.jsonl").symlink_to(external_root / "governance" / "external.jsonl")

    payload = src._take_curated_snapshot(
        local_root,
        tmp_path / "recovery",
        apply=True,
        state={},
        cooldown_seconds=0.0,
        project_root=tmp_path,
    )

    selected = {row["rel_path"]: row for row in payload["selected_paths"]}
    assert payload["ok"] is True
    assert selected["governance"]["eligible"] is False
    assert selected["governance"]["skip_reason"] == "outside_local_fallback_root"
    assert not (tmp_path / "recovery" / "latest" / "governance" / "external.jsonl").exists()
    assert not (tmp_path / "recovery" / "latest" / "logs" / "external.jsonl").exists()
    assert "logs/external.jsonl" in payload["unsafe_skipped_paths"]
    assert (tmp_path / "recovery" / "latest" / "logs" / "local.log").exists()


def test_take_curated_snapshot_cleans_abandoned_staging(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_LOGS_RECOVERY_MIN_FREE_AFTER_SNAPSHOT_GB", "0")
    local_root = tmp_path / "local_fallback_storage"
    (local_root / "logs").mkdir(parents=True)
    (local_root / "logs" / "local.log").write_text("local", encoding="utf-8")
    abandoned = tmp_path / "recovery" / ".latest_staging_123"
    abandoned.mkdir(parents=True)
    (abandoned / "partial.bin").write_bytes(b"partial")

    payload = src._take_curated_snapshot(
        local_root,
        tmp_path / "recovery",
        apply=True,
        state={},
        cooldown_seconds=0.0,
        project_root=tmp_path,
    )

    assert payload["workspace_cleanup"]["deleted_count"] == 1
    assert not abandoned.exists()


def test_take_curated_snapshot_requires_capacity_for_reserve(monkeypatch, tmp_path: Path) -> None:
    local_root = tmp_path / "local_fallback_storage"
    (local_root / "logs").mkdir(parents=True)
    (local_root / "logs" / "local.log").write_bytes(b"x" * 1024)
    monkeypatch.setenv("BOT_LOGS_RECOVERY_MIN_FREE_AFTER_SNAPSHOT_GB", "64")
    monkeypatch.setattr(src.shutil, "disk_usage", lambda _path: type("Usage", (), {"free": 1024})())

    payload = src._take_curated_snapshot(
        local_root,
        tmp_path / "recovery",
        apply=True,
        state={},
        cooldown_seconds=0.0,
        project_root=tmp_path,
    )

    assert payload["ok"] is False
    assert payload["skipped_reason"] == "insufficient_local_capacity"
    assert payload["capacity_preflight"]["sufficient"] is False
    assert not (tmp_path / "recovery" / "latest").exists()


def test_sync_storage_target_override_writes_single_source_of_truth(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    probe = {
        "mount_root": "/Volumes/BOT_LOGS",
        "candidate_mount_roots": ["/Volumes/BOT_LOGS"],
        "target_volume_name": "BOT_LOGS",
        "target_volume_uuid": "uuid-123",
        "target_volume_device_identifier": "disk9s1",
    }

    payload = src._sync_storage_target_override(project_root, probe, apply=True)

    text = Path(payload["path"]).read_text(encoding="utf-8")
    assert payload["ok"] is True
    assert "BOT_LOGS_EXTERNAL_MOUNT=/Volumes/BOT_LOGS" in text
    assert "BOT_LOGS_EXTERNAL_VOLUME_UUID=uuid-123" in text
