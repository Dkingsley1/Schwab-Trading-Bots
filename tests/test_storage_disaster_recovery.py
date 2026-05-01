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


def test_take_curated_snapshot_copies_only_important_paths(tmp_path: Path) -> None:
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
