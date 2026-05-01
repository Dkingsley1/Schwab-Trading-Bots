from pathlib import Path

from core import storage_mounts


def test_resolve_external_storage_prefers_existing_candidate_mount(monkeypatch, tmp_path):
    missing_mount = tmp_path / "BOT_LOGS"
    video_mount = tmp_path / "VIDEO"
    external_root = video_mount / "schwab_trading_bot"
    external_root.mkdir(parents=True)

    monkeypatch.setenv("BOT_LOGS_EXTERNAL_MOUNT", str(missing_mount))
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES", f"{missing_mount},{video_mount}")
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot")

    resolution = storage_mounts.resolve_external_storage()

    assert resolution.mount_root == video_mount
    assert resolution.external_root == external_root
    assert resolution.matched_mount_root == video_mount
    assert resolution.match_reason == "candidate_project_root_exists"


def test_resolve_external_storage_uses_existing_configured_project_root(monkeypatch, tmp_path):
    configured_root = tmp_path / "custom_external_root"
    configured_root.mkdir(parents=True)

    monkeypatch.setenv("BOT_LOGS_EXTERNAL_MOUNT", str(tmp_path / "BOT_LOGS"))
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(configured_root))
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES", raising=False)

    resolution = storage_mounts.resolve_external_storage()

    assert resolution.external_root == configured_root
    assert resolution.matched_mount_root == resolution.mount_root
    assert resolution.match_reason == "configured_project_root_exists"


def test_external_mount_candidates_default_to_configured_mount(monkeypatch):
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES", raising=False)
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)

    candidates = storage_mounts.external_mount_candidates()

    assert candidates == (Path("/Volumes/BOT_LOGS"),)
