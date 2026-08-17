from pathlib import Path

from scripts.ops import bot_logs_finder_sync


def test_sync_shortcuts_creates_alias_and_desktop_shortcut(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    logs_path = project_root / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    alias_path = tmp_path / "home" / "bot_logs"
    desktop_path = tmp_path / "home" / "Desktop" / "Bot Logs"

    payload = bot_logs_finder_sync.sync_shortcuts(
        project_root,
        alias_path=alias_path,
        desktop_path=desktop_path,
        create_desktop_shortcut=True,
    )

    assert payload["ok"] is True
    assert alias_path.is_symlink()
    assert desktop_path.is_symlink()
    assert alias_path.resolve() == logs_path.resolve()
    assert desktop_path.resolve() == logs_path.resolve()


def test_sync_shortcuts_can_skip_desktop_shortcut(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    logs_path = project_root / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    alias_path = tmp_path / "home" / "bot_logs"
    desktop_path = tmp_path / "home" / "Desktop" / "Bot Logs"

    payload = bot_logs_finder_sync.sync_shortcuts(
        project_root,
        alias_path=alias_path,
        desktop_path=desktop_path,
        create_desktop_shortcut=False,
    )

    assert payload["ok"] is True
    assert alias_path.is_symlink()
    assert not desktop_path.exists()
    assert payload["desktop_shortcut"]["enabled"] is False
