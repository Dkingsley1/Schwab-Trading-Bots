from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _freshness_norm(age_seconds: float) -> float:
    if age_seconds <= 0.0:
        return 1.0
    if age_seconds >= 24.0 * 3600.0:
        return 0.0
    return _clamp01(1.0 - (age_seconds / (24.0 * 3600.0)))


def _record_watermark(
    *,
    project_root: Path | None,
    collector_key: str,
    source_name: str,
    entity_key: str,
    watermark_value: str,
    etag: str,
    payload_sha256: str,
    metadata: Mapping[str, Any],
) -> None:
    if project_root is None:
        return
    if str(os.getenv("COLLECTOR_TRANSPORT_SKIP_WATERMARKS", "0") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    try:
        from scripts import ops_data_plane

        with ops_data_plane.connect(Path(project_root), quick_check=False, timeout_seconds=2.0) as conn:
            ops_data_plane.record_watermark(
                conn,
                collector_key=str(collector_key),
                source_name=str(source_name),
                entity_key=ops_data_plane.normalize_entity_key(Path(project_root), entity_key, namespace="source"),
                watermark_type="http_fetch",
                watermark_value=str(watermark_value),
                etag=str(etag or ""),
                payload_sha256=str(payload_sha256 or ""),
                metadata=dict(metadata or {}),
            )
    except Exception:
        return


def fetch_text(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    accept: str = "*/*",
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    body: bytes | None = None,
    retries: int = 2,
    backoff_seconds: float = 0.75,
    jitter_seconds: float = 0.35,
    etag: str = "",
    collector_key: str = "",
    source_name: str = "",
    entity_key: str = "",
    project_root: Path | None = None,
    source_confidence_norm: float = 0.9,
    schema_confidence_norm: float = 0.95,
) -> dict[str, Any]:
    req_headers = {
        "User-Agent": str(user_agent or "").strip() or "schwab-trading-bot/1.0",
        "Accept": str(accept or "*/*"),
    }
    if etag:
        req_headers["If-None-Match"] = str(etag)
    for key, value in dict(headers or {}).items():
        if str(key).strip():
            req_headers[str(key)] = str(value)

    attempts = 0
    last_error = ""
    while attempts <= max(int(retries), 0):
        attempts += 1
        started = time.time()
        try:
            req = Request(url=str(url), method=str(method or "GET").upper(), headers=req_headers, data=body)
            with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
                raw = resp.read().decode("utf-8", "replace")
                fetched_utc = _now_utc()
                payload_sha256 = _sha256_text(raw)
                response_etag = str(resp.headers.get("ETag") or "")
                duration_ms = round((time.time() - started) * 1000.0, 3)
                metadata = {
                    "url": str(url),
                    "status_code": int(getattr(resp, "status", 200) or 200),
                    "attempt_count": int(attempts),
                    "duration_ms": duration_ms,
                    "size_bytes": len(raw.encode("utf-8")),
                    "source_confidence_norm": _clamp01(source_confidence_norm),
                    "schema_confidence_norm": _clamp01(schema_confidence_norm),
                }
                _record_watermark(
                    project_root=project_root,
                    collector_key=collector_key,
                    source_name=source_name,
                    entity_key=entity_key or url,
                    watermark_value=fetched_utc,
                    etag=response_etag,
                    payload_sha256=payload_sha256,
                    metadata=metadata,
                )
                return {
                    "ok": True,
                    "url": str(url),
                    "text": raw,
                    "status_code": int(getattr(resp, "status", 200) or 200),
                    "etag": response_etag,
                    "fetched_utc": fetched_utc,
                    "attempt_count": int(attempts),
                    "duration_ms": duration_ms,
                    "payload_sha256": payload_sha256,
                    "size_bytes": len(raw.encode("utf-8")),
                    "source_confidence_norm": _clamp01(source_confidence_norm),
                    "schema_confidence_norm": _clamp01(schema_confidence_norm),
                    "freshness_norm": _freshness_norm(0.0),
                    "provenance": metadata,
                }
        except HTTPError as exc:
            if int(getattr(exc, "code", 0) or 0) == 304:
                return {
                    "ok": True,
                    "url": str(url),
                    "text": "",
                    "status_code": 304,
                    "etag": str(etag or ""),
                    "fetched_utc": _now_utc(),
                    "attempt_count": int(attempts),
                    "duration_ms": round((time.time() - started) * 1000.0, 3),
                    "payload_sha256": "",
                    "size_bytes": 0,
                    "source_confidence_norm": _clamp01(source_confidence_norm),
                    "schema_confidence_norm": _clamp01(schema_confidence_norm),
                    "freshness_norm": _freshness_norm(0.0),
                    "provenance": {
                        "url": str(url),
                        "status_code": 304,
                        "attempt_count": int(attempts),
                        "cache_revalidated": True,
                    },
                }
            last_error = str(exc)
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = str(exc)
        if attempts <= max(int(retries), 0):
            delay = max(float(backoff_seconds), 0.1) * attempts
            delay += random.uniform(0.0, max(float(jitter_seconds), 0.0))
            time.sleep(min(delay, 5.0))
    return {
        "ok": False,
        "url": str(url),
        "text": "",
        "status_code": None,
        "etag": str(etag or ""),
        "fetched_utc": _now_utc(),
        "attempt_count": int(attempts),
        "duration_ms": None,
        "payload_sha256": "",
        "size_bytes": 0,
        "source_confidence_norm": _clamp01(source_confidence_norm),
        "schema_confidence_norm": _clamp01(schema_confidence_norm),
        "freshness_norm": 0.0,
        "error": last_error,
        "provenance": {
            "url": str(url),
            "attempt_count": int(attempts),
            "error": last_error,
        },
    }


def fetch_json(**kwargs: Any) -> dict[str, Any]:
    result = fetch_text(accept="application/json", **kwargs)
    if not result.get("ok", False):
        return result
    if int(result.get("status_code") or 0) == 304:
        result["json"] = None
        return result
    try:
        result["json"] = json.loads(str(result.get("text") or ""))
        return result
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}:{exc}"
        return result


def attach_collection_confidence(
    row: Mapping[str, Any],
    *,
    source_confidence_norm: float,
    schema_confidence_norm: float,
    freshness_norm: float,
    fetched_utc: str = "",
) -> dict[str, Any]:
    out = dict(row)
    out["source_confidence_norm"] = _clamp01(source_confidence_norm)
    out["schema_confidence_norm"] = _clamp01(schema_confidence_norm)
    out["freshness_norm"] = _clamp01(freshness_norm)
    if fetched_utc:
        out.setdefault("fetched_utc", str(fetched_utc))
    return out


class AsyncCollectorHTTPClient:
    async def fetch_text(self, **kwargs: Any) -> dict[str, Any]:
        try:
            import aiohttp
        except Exception as exc:
            return {
                "ok": False,
                "error": f"aiohttp_unavailable:{exc}",
                "attempt_count": 0,
                "status_code": None,
                "text": "",
                "fetched_utc": _now_utc(),
            }

        url = str(kwargs.get("url") or "")
        timeout = max(float(kwargs.get("timeout") or 25.0), 1.0)
        headers = {
            "User-Agent": str(kwargs.get("user_agent") or "schwab-trading-bot/1.0"),
            "Accept": str(kwargs.get("accept") or "*/*"),
        }
        for key, value in dict(kwargs.get("headers") or {}).items():
            headers[str(key)] = str(value)
        started = time.time()
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=timeout) as resp:
                text = await resp.text()
                return {
                    "ok": True,
                    "url": url,
                    "text": text,
                    "status_code": int(resp.status),
                    "fetched_utc": _now_utc(),
                    "attempt_count": 1,
                    "duration_ms": round((time.time() - started) * 1000.0, 3),
                    "payload_sha256": _sha256_text(text),
                    "size_bytes": len(text.encode("utf-8")),
                }
