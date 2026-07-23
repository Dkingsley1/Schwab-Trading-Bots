from __future__ import annotations

import builtins
from pathlib import Path

from core import collector_transport
from scripts import ops_data_plane


def test_record_watermark_skip_env_avoids_sqlite_import(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COLLECTOR_TRANSPORT_SKIP_WATERMARKS", "1")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "scripts":
            raise AssertionError("watermark skip should avoid importing ops_data_plane")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    collector_transport._record_watermark(
        project_root=tmp_path,
        collector_key="test_collector",
        source_name="unit_source",
        entity_key="AAPL",
        watermark_value="2026-06-15T00:00:00+00:00",
        etag="",
        payload_sha256="abc123",
        metadata={"ok": True},
    )


def test_record_watermark_uses_fast_ops_data_plane_connect(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_connect(project_root: Path, **kwargs):
        captured["project_root"] = project_root
        captured["kwargs"] = kwargs
        return FakeConn()

    def fake_record_watermark(_conn, **kwargs):
        captured["watermark"] = kwargs

    monkeypatch.setattr(ops_data_plane, "connect", fake_connect)
    monkeypatch.setattr(ops_data_plane, "record_watermark", fake_record_watermark)
    monkeypatch.setattr(ops_data_plane, "normalize_entity_key", lambda _root, key, namespace="": f"{namespace}/{key}")

    collector_transport._record_watermark(
        project_root=tmp_path,
        collector_key="test_collector",
        source_name="unit_source",
        entity_key="AAPL",
        watermark_value="2026-06-29T00:00:00+00:00",
        etag="",
        payload_sha256="abc123",
        metadata={"ok": True},
    )

    assert captured["project_root"] == tmp_path
    assert captured["kwargs"] == {"quick_check": False, "timeout_seconds": 2.0}
    assert captured["watermark"]["entity_key"] == "source/AAPL"
