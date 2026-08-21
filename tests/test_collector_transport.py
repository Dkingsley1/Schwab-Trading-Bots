from __future__ import annotations

import asyncio
import builtins
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

from core import collector_transport
from scripts import ops_data_plane


class _Response:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict[str, str] | None = None):
        self.body = body
        self.status = status
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value
        self.read_size = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        self.read_size = size
        return self.body if size < 0 else self.body[:size]


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


def test_fetch_text_emits_bounded_signed_route_receipt_and_redacts_query(monkeypatch) -> None:
    response = _Response(b"payload", headers={"ETag": "v2"})
    monkeypatch.setenv("COLLECTOR_TRANSPORT_SKIP_WATERMARKS", "1")
    monkeypatch.setattr(collector_transport, "urlopen", lambda *_args, **_kwargs: response)

    result = collector_transport.fetch_text(
        "https://example.test/feed?api_key=secret",
        user_agent="test",
        timeout=2.0,
        retries=0,
        max_response_bytes=1024,
        route_id="route_alpha",
        capability_ids=["ohlcv_bars", "source_freshness"],
    )

    assert result["ok"] is True
    assert result["url"] == "https://example.test/feed"
    assert "secret" not in str(result)
    assert result["transport_contract_version"] == "collector_transport_v2"
    assert result["route_id"] == "route_alpha"
    assert result["capability_ids"] == ["ohlcv_bars", "source_freshness"]
    assert len(result["transport_receipt_sha256"]) == 64
    assert response.read_size == 1025


def test_fetch_text_does_not_retry_permanent_http_failure(monkeypatch) -> None:
    calls = 0

    def denied(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise HTTPError(
            "https://example.test/private?token=secret",
            401,
            "Unauthorized",
            Message(),
            None,
        )

    monkeypatch.setenv("COLLECTOR_TRANSPORT_SKIP_DEAD_LETTERS", "1")
    monkeypatch.setattr(collector_transport, "urlopen", denied)
    monkeypatch.setattr(collector_transport.time, "sleep", lambda _seconds: None)

    result = collector_transport.fetch_text(
        "https://example.test/private?token=secret",
        user_agent="test",
        timeout=2.0,
        retries=4,
    )

    assert calls == 1
    assert result["ok"] is False
    assert result["status_code"] == 401
    assert result["error_class"] == "http_401"
    assert result["retry_exhausted"] is False
    assert "secret" not in str(result)


def test_fetch_text_retries_transient_http_failure_and_respects_retry_after(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def flaky(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            headers = Message()
            headers["Retry-After"] = "2"
            raise HTTPError("https://example.test/feed", 429, "Limited", headers, None)
        return _Response(b"ok")

    monkeypatch.setenv("COLLECTOR_TRANSPORT_SKIP_WATERMARKS", "1")
    monkeypatch.setattr(collector_transport, "urlopen", flaky)
    monkeypatch.setattr(collector_transport.time, "sleep", sleeps.append)

    result = collector_transport.fetch_text(
        "https://example.test/feed",
        user_agent="test",
        timeout=2.0,
        retries=2,
    )

    assert calls == 2
    assert sleeps == [2.0]
    assert result["ok"] is True
    assert result["attempt_count"] == 2


def test_fetch_text_fails_closed_on_oversized_payload_without_retry(monkeypatch) -> None:
    calls = 0

    def oversized(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _Response(b"x" * 2048, headers={"Content-Length": "2048"})

    monkeypatch.setenv("COLLECTOR_TRANSPORT_SKIP_DEAD_LETTERS", "1")
    monkeypatch.setattr(collector_transport, "urlopen", oversized)
    monkeypatch.setattr(collector_transport.time, "sleep", lambda _seconds: None)

    result = collector_transport.fetch_text(
        "https://example.test/large",
        user_agent="test",
        timeout=2.0,
        retries=3,
        max_response_bytes=1024,
    )

    assert calls == 1
    assert result["ok"] is False
    assert result["error_class"] == "response_too_large"
    assert result["retry_exhausted"] is False


def test_async_transport_reuses_canonical_signed_contract(monkeypatch) -> None:
    calls: list[str] = []

    def fake_fetch_json(**kwargs):
        calls.append(str(kwargs["url"]))
        return {
            "ok": True,
            "url": str(kwargs["url"]),
            "transport_contract_version": "collector_transport_v2",
        }

    monkeypatch.setattr(collector_transport, "fetch_json", fake_fetch_json)
    client = collector_transport.AsyncCollectorHTTPClient(max_concurrency=2)
    rows = asyncio.run(
        client.fetch_many(
            [
                {"url": "https://one.test", "user_agent": "test", "timeout": 1},
                {"url": "https://two.test", "user_agent": "test", "timeout": 1},
            ]
        )
    )

    assert calls == ["https://one.test", "https://two.test"]
    assert [row["transport_contract_version"] for row in rows] == [
        "collector_transport_v2",
        "collector_transport_v2",
    ]
