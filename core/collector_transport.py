from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


TRANSPORT_CONTRACT_VERSION = "collector_transport_v2"
EVENT_TIME_CONTRACT_VERSION = "collector_event_time_v1"
TRANSIENT_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_RETRY_AFTER_SECONDS = 30.0


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _redacted_url(url: str) -> str:
    try:
        parts = urlsplit(str(url or ""))
    except ValueError:
        return "invalid_url"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _request_id(
    *,
    url: str,
    collector_key: str,
    source_name: str,
    entity_key: str,
) -> str:
    material = {
        "timestamp_ns": time.time_ns(),
        "pid": os.getpid(),
        "url": _redacted_url(url),
        "collector_key": collector_key,
        "source_name": source_name,
        "entity_key": entity_key,
    }
    return f"fetch_{_canonical_hash(material)[:20]}"


def _retry_after_seconds(headers: Any) -> float | None:
    try:
        raw = str(headers.get("Retry-After") or "").strip()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return min(max(float(raw), 0.0), MAX_RETRY_AFTER_SECONDS)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(raw)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = (retry_at.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
        return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)
    except (TypeError, ValueError, OverflowError):
        return None


def _transport_receipt(
    payload: dict[str, Any],
    *,
    request_id: str,
    route_id: str,
    capability_ids: Sequence[str],
    redacted_url: str,
) -> dict[str, Any]:
    payload["transport_contract_version"] = TRANSPORT_CONTRACT_VERSION
    payload["request_id"] = request_id
    payload["route_id"] = str(route_id or "")
    payload["capability_ids"] = sorted(
        {str(value) for value in capability_ids if str(value).strip()}
    )
    payload["url"] = redacted_url
    receipt_material = {
        "transport_contract_version": TRANSPORT_CONTRACT_VERSION,
        "request_id": request_id,
        "route_id": payload["route_id"],
        "capability_ids": payload["capability_ids"],
        "url": redacted_url,
        "ok": bool(payload.get("ok", False)),
        "status_code": payload.get("status_code"),
        "fetched_utc": str(payload.get("fetched_utc") or ""),
        "attempt_count": int(payload.get("attempt_count", 0) or 0),
        "payload_sha256": str(payload.get("payload_sha256") or ""),
        "size_bytes": int(payload.get("size_bytes", 0) or 0),
        "error_class": str(payload.get("error_class") or ""),
    }
    payload["transport_receipt_sha256"] = _canonical_hash(receipt_material)
    return payload


def _freshness_norm(age_seconds: float) -> float:
    if age_seconds <= 0.0:
        return 1.0
    if age_seconds >= 24.0 * 3600.0:
        return 0.0
    return _clamp01(1.0 - (age_seconds / (24.0 * 3600.0)))


def qualify_transport_event(
    transport_result: Mapping[str, Any],
    *,
    guard: Any,
    stream_id: str,
    source_event_time_utc: str,
    event_id: str = "",
    event_payload: Any | None = None,
) -> dict[str, Any]:
    """Attach event-time usability without conflating it with HTTP transport success.

    Call this after a collector parses the source event timestamp. A successful HTTP
    fetch can still be unusable when the event is late, future-skewed, or conflicts
    with a prior event identity.
    """
    result = dict(transport_result or {})
    active_event_id = str(event_id or result.get("request_id") or "").strip()
    if not result.get("ok", False):
        result["event_time_contract_version"] = EVENT_TIME_CONTRACT_VERSION
        result["event_time_usable"] = False
        result["event_time"] = {
            "accepted": False,
            "disposition": "transport_unavailable",
            "reason": "transport_not_successful",
        }
        return result
    if guard is None or not hasattr(guard, "ingest"):
        raise ValueError("a stateful EventTimeGuard is required")
    decision = guard.ingest(
        stream_id=str(stream_id or "").strip(),
        event_id=active_event_id,
        event_time_utc=source_event_time_utc,
        observed_at_utc=str(result.get("fetched_utc") or _now_utc()),
        payload=(
            event_payload
            if event_payload is not None
            else {"payload_sha256": str(result.get("payload_sha256") or "")}
        ),
    )
    result["event_time_contract_version"] = EVENT_TIME_CONTRACT_VERSION
    result["event_time_usable"] = bool(decision.get("accepted", False))
    result["event_time"] = decision
    result["quarantined"] = bool(decision.get("disposition") == "quarantine")
    return result


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


def _record_transport_failure(
    *,
    project_root: Path | None,
    collector_key: str,
    route_id: str,
    redacted_url: str,
    request_id: str,
    error_class: str,
    error_message: str,
    attempts: int,
) -> None:
    if project_root is None:
        return
    if str(
        os.getenv("COLLECTOR_TRANSPORT_SKIP_DEAD_LETTERS", "0") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}:
        return
    try:
        from scripts import ops_data_plane

        with ops_data_plane.connect(
            Path(project_root), quick_check=False, timeout_seconds=2.0
        ) as conn:
            ops_data_plane.record_dead_letter(
                conn,
                lane="collector_transport",
                source_rel=redacted_url,
                line_no=0,
                offset_bytes=0,
                error_class=str(error_class or "collector_transport_failure"),
                error_message=str(error_message or "")[:512],
                raw_payload="",
                run_id=request_id,
                metadata={
                    "collector_key": str(collector_key or ""),
                    "route_id": str(route_id or ""),
                    "attempt_count": int(attempts),
                    "transport_contract_version": TRANSPORT_CONTRACT_VERSION,
                },
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
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    request_id: str = "",
    route_id: str = "",
    capability_ids: Sequence[str] = (),
) -> dict[str, Any]:
    safe_url = _redacted_url(url)
    active_request_id = str(request_id or "").strip() or _request_id(
        url=url,
        collector_key=collector_key,
        source_name=source_name,
        entity_key=entity_key,
    )
    response_limit = max(int(max_response_bytes), 1024)
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
    last_error_class = ""
    last_status_code: int | None = None
    while attempts <= max(int(retries), 0):
        attempts += 1
        started = time.time()
        retryable = True
        retry_after: float | None = None
        try:
            req = Request(url=str(url), method=str(method or "GET").upper(), headers=req_headers, data=body)
            with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
                content_length = int(resp.headers.get("Content-Length") or 0)
                if content_length > response_limit:
                    raise ValueError(
                        f"response_too_large:{content_length}>{response_limit}"
                    )
                raw_bytes = resp.read(response_limit + 1)
                if len(raw_bytes) > response_limit:
                    raise ValueError(
                        f"response_too_large:{len(raw_bytes)}>{response_limit}"
                    )
                raw = raw_bytes.decode("utf-8", "replace")
                fetched_utc = _now_utc()
                payload_sha256 = _sha256_text(raw)
                response_etag = str(resp.headers.get("ETag") or "")
                duration_ms = round((time.time() - started) * 1000.0, 3)
                metadata = {
                    "url": safe_url,
                    "status_code": int(getattr(resp, "status", 200) or 200),
                    "attempt_count": int(attempts),
                    "duration_ms": duration_ms,
                    "size_bytes": len(raw_bytes),
                    "source_confidence_norm": _clamp01(source_confidence_norm),
                    "schema_confidence_norm": _clamp01(schema_confidence_norm),
                    "request_id": active_request_id,
                    "route_id": str(route_id or ""),
                    "capability_ids": sorted(
                        {
                            str(value)
                            for value in capability_ids
                            if str(value).strip()
                        }
                    ),
                    "response_size_limit_bytes": response_limit,
                }
                result = _transport_receipt(
                    {
                        "ok": True,
                        "url": safe_url,
                        "text": raw,
                        "status_code": int(getattr(resp, "status", 200) or 200),
                        "etag": response_etag,
                        "fetched_utc": fetched_utc,
                        "attempt_count": int(attempts),
                        "duration_ms": duration_ms,
                        "payload_sha256": payload_sha256,
                        "size_bytes": len(raw_bytes),
                        "source_confidence_norm": _clamp01(source_confidence_norm),
                        "schema_confidence_norm": _clamp01(schema_confidence_norm),
                        "freshness_norm": _freshness_norm(0.0),
                        "response_size_limit_bytes": response_limit,
                        "provenance": metadata,
                    },
                    request_id=active_request_id,
                    route_id=route_id,
                    capability_ids=capability_ids,
                    redacted_url=safe_url,
                )
                metadata["transport_receipt_sha256"] = result[
                    "transport_receipt_sha256"
                ]
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
                return result
        except HTTPError as exc:
            last_status_code = int(getattr(exc, "code", 0) or 0)
            if last_status_code == 304:
                return _transport_receipt(
                    {
                        "ok": True,
                        "url": safe_url,
                        "text": "",
                        "status_code": 304,
                        "etag": str(etag or ""),
                        "fetched_utc": _now_utc(),
                        "attempt_count": int(attempts),
                        "duration_ms": round(
                            (time.time() - started) * 1000.0, 3
                        ),
                        "payload_sha256": "",
                        "size_bytes": 0,
                        "source_confidence_norm": _clamp01(
                            source_confidence_norm
                        ),
                        "schema_confidence_norm": _clamp01(
                            schema_confidence_norm
                        ),
                        "freshness_norm": _freshness_norm(0.0),
                        "response_size_limit_bytes": response_limit,
                        "provenance": {
                            "url": safe_url,
                            "status_code": 304,
                            "attempt_count": int(attempts),
                            "cache_revalidated": True,
                            "request_id": active_request_id,
                            "route_id": str(route_id or ""),
                        },
                    },
                    request_id=active_request_id,
                    route_id=route_id,
                    capability_ids=capability_ids,
                    redacted_url=safe_url,
                )
            last_error_class = f"http_{last_status_code}"
            last_error = last_error_class
            retryable = last_status_code in TRANSIENT_HTTP_STATUS_CODES
            retry_after = _retry_after_seconds(getattr(exc, "headers", {}))
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            last_error_class = (
                "response_too_large"
                if isinstance(exc, ValueError)
                and str(exc).startswith("response_too_large:")
                else type(exc).__name__.lower()
            )
            last_error = (
                str(exc)
                if last_error_class == "response_too_large"
                else last_error_class
            )
            retryable = last_error_class != "response_too_large"
        if attempts <= max(int(retries), 0) and retryable:
            delay = (
                retry_after
                if retry_after is not None
                else max(float(backoff_seconds), 0.1) * attempts
                + random.uniform(0.0, max(float(jitter_seconds), 0.0))
            )
            time.sleep(min(max(delay, 0.0), MAX_RETRY_AFTER_SECONDS))
            continue
        break
    _record_transport_failure(
        project_root=project_root,
        collector_key=collector_key,
        route_id=route_id,
        redacted_url=safe_url,
        request_id=active_request_id,
        error_class=last_error_class,
        error_message=last_error,
        attempts=attempts,
    )
    return _transport_receipt(
        {
            "ok": False,
            "url": safe_url,
            "text": "",
            "status_code": last_status_code,
            "etag": str(etag or ""),
            "fetched_utc": _now_utc(),
            "attempt_count": int(attempts),
            "duration_ms": None,
            "payload_sha256": "",
            "size_bytes": 0,
            "source_confidence_norm": _clamp01(source_confidence_norm),
            "schema_confidence_norm": _clamp01(schema_confidence_norm),
            "freshness_norm": 0.0,
            "response_size_limit_bytes": response_limit,
            "error": last_error,
            "error_class": last_error_class,
            "retry_exhausted": bool(
                retryable and attempts > max(int(retries), 0)
            ),
            "provenance": {
                "url": safe_url,
                "attempt_count": int(attempts),
                "error": last_error,
                "error_class": last_error_class,
                "request_id": active_request_id,
                "route_id": str(route_id or ""),
            },
        },
        request_id=active_request_id,
        route_id=route_id,
        capability_ids=capability_ids,
        redacted_url=safe_url,
    )


def fetch_json(**kwargs: Any) -> dict[str, Any]:
    result = fetch_text(accept="application/json", **kwargs)
    if not result.get("ok", False):
        return result
    if int(result.get("status_code") or 0) == 304:
        result["json"] = None
        return result
    try:
        result["json"] = json.loads(str(result.get("text") or ""))
    except Exception as exc:
        result["ok"] = False
        result["error_class"] = "json_decode_error"
        result["error"] = f"{type(exc).__name__}:{exc}"
    return _transport_receipt(
        result,
        request_id=str(result.get("request_id") or ""),
        route_id=str(result.get("route_id") or ""),
        capability_ids=list(result.get("capability_ids") or []),
        redacted_url=str(result.get("url") or ""),
    )


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
    """Bounded async facade over the canonical signed collector transport."""

    def __init__(self, *, max_concurrency: int = 8) -> None:
        self.max_concurrency = max(int(max_concurrency), 1)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    async def fetch_text(self, **kwargs: Any) -> dict[str, Any]:
        async with self._semaphore:
            return await asyncio.to_thread(fetch_text, **kwargs)

    async def fetch_json(self, **kwargs: Any) -> dict[str, Any]:
        async with self._semaphore:
            return await asyncio.to_thread(fetch_json, **kwargs)

    async def fetch_many(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        response_kind: str = "json",
    ) -> list[dict[str, Any]]:
        method = self.fetch_json if response_kind == "json" else self.fetch_text
        return list(
            await asyncio.gather(
                *(method(**dict(request)) for request in requests)
            )
        )
