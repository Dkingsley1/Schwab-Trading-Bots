import json
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
    assert stale_file.resolve(strict=True) == target.resolve(strict=True)
    assert payload["top_rows"][0]["source_replaced_with_symlink"] is True
    manifest = Path(payload["manifest_path"])
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_replaced_with_symlink"] is True


def test_video_cold_archive_subtree_is_allowed_only_when_explicitly_enabled(monkeypatch) -> None:
    root = Path("/Volumes/VIDEO/schwab_trading_bot_cold")

    monkeypatch.delenv("BOT_ALLOW_VIDEO_COLD_ARCHIVE", raising=False)
    assert src._is_protected_volume(Path("/Volumes/VIDEO")) is True
    assert src._is_protected_volume(root / "deep_cold" / "file.gz") is True

    monkeypatch.setenv("BOT_ALLOW_VIDEO_COLD_ARCHIVE", "1")
    monkeypatch.setenv("BOT_VIDEO_COLD_ARCHIVE_ROOT", str(root))
    assert src._is_protected_volume(Path("/Volumes/VIDEO")) is True
    assert src._is_protected_volume(root / "deep_cold" / "file.gz") is False
