#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time as time_mod
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python
from scripts.ops import ingestion_storage_governor as governor_src
from scripts.ops.long_runtime_common import us_equity_market_holiday


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "external_backlog_drain_latest.json"
SERVICE_REQUEST_PATH = PROJECT_ROOT / "governance" / "health" / "sql_link_service_request_latest.json"
LOCAL_TZ = ZoneInfo("America/New_York")
OFF_HOURS_START = time(16, 15)
OFF_HOURS_END = time(9, 20)
SQL_WRITER_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "jsonl_sql_writer.lock"
DEFAULT_DRAIN_SHARDS = [
    "health_fast",
    "trading_fast",
    "crypto_trading_fast",
    "crypto_trading",
    "runtime",
    "crypto_runtime",
    "aggressive_trading",
    "trading",
    "governance",
    "support_watchdog",
    "risk_support",
    "crypto_governance",
    "data",
    "explanations",
    "crypto_explanations",
    "shadow_attribution",
    "crypto_shadow_attribution",
]
CORE_FOCUS_MIN_PENDING_LINES = 30_000
CORE_FOCUS_MIN_TOP3_LINES = 40_000
CORE_FOCUS_MIN_SHARE = 0.65
MASSIVE_FOCUS_MIN_FILE_PENDING_LINES = 100_000
MASSIVE_FOCUS_MIN_TOP3_LINES = 250_000
DRAIN_RECOMMEND_MIN_TOTAL_PENDING_LINES = 3_000
DRAIN_RECOMMEND_MIN_DEFERRED_PENDING_LINES = 1_000
DRAIN_RECOMMEND_MIN_SUPPORT_PENDING_LINES = 500
SPARSE_LARGE_JSONL_PENDING_BYTES_FLOOR = 64 * 1024 * 1024
SPARSE_LARGE_DECISION_MAX_BYTES_PER_FILE = 128 * 1024 * 1024
SPARSE_LARGE_DECISION_SQLITE_BATCH_MAX_BYTES = 32 * 1024 * 1024


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_json_output(text: str) -> dict[str, Any]:
    for line in reversed([raw.strip() for raw in str(text or "").splitlines() if raw.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _preservable_service_request(path: Path, *, now_utc: datetime) -> dict[str, Any]:
    payload = _load_json(path)
    if not payload or payload.get("active") is False:
        return {}
    if str(payload.get("request_kind") or "") != "backpressure_drainer_fleet":
        return {}
    expires_utc = _parse_iso_utc(payload.get("expires_utc"))
    if expires_utc is not None and expires_utc <= now_utc:
        return {}
    env = payload.get("env_overrides") if isinstance(payload.get("env_overrides"), dict) else {}
    if not str(env.get("SQL_LINK_SERVICE_SHARDS") or "").strip():
        return {}
    preserved = dict(payload)
    preserved["preserved_existing_request"] = True
    preserved["preserved_by"] = "external_backlog_drain"
    return preserved


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _age_bucket(age_seconds: float) -> str:
    age = max(float(age_seconds), 0.0)
    if age < 30 * 60:
        return "fresh_lt_30m"
    if age < 2 * 60 * 60:
        return "aging_lt_2h"
    if age < 6 * 60 * 60:
        return "stale_lt_6h"
    if age < 24 * 60 * 60:
        return "stale_lt_24h"
    return "cold_gte_24h"


def _lock_owner_pid(lock_path: Path) -> int | None:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    for token in raw.split():
        if not token.startswith("pid="):
            continue
        try:
            return int(token.split("=", 1)[1])
        except Exception:
            return None
    return None


def _off_hours_window(now_utc: datetime) -> dict[str, Any]:
    local_now = now_utc.astimezone(LOCAL_TZ)
    local_clock = local_now.timetz().replace(tzinfo=None)
    is_weekend = local_now.weekday() >= 5
    holiday_name = us_equity_market_holiday(local_now.date())
    market_holiday = bool(holiday_name)
    active = bool(is_weekend or market_holiday or local_clock >= OFF_HOURS_START or local_clock < OFF_HOURS_END)
    return {
        "active": active,
        "is_weekend": is_weekend,
        "market_holiday": market_holiday,
        "market_holiday_name": holiday_name,
        "timezone": "America/New_York",
        "local_time": local_now.isoformat(),
        "window_start_local": OFF_HOURS_START.strftime("%H:%M"),
        "window_end_local": OFF_HOURS_END.strftime("%H:%M"),
        "label": "off_hours" if active else "market_hours",
    }


def _run_json_command(
    cmd: list[str],
    *,
    cwd: Path,
    payload_path: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=timeout_seconds,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    payload = _parse_json_output(stdout)
    if not payload and payload_path is not None:
        payload = _load_json(payload_path)
    if timed_out:
        payload = {**payload, "ok": False, "reason": "timeout", "timed_out": True}
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(cmd),
        "rc": rc,
        "duration_ms": duration_ms,
        "payload": payload,
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]),
        "timed_out": timed_out,
    }


def _step_status(result: dict[str, Any], *, nonfatal_reasons: set[str] | None = None) -> str:
    if bool(result.get("timed_out", False)):
        return "busy"
    if int(result.get("rc", 1)) != 0:
        return "error"
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    reason = str(payload.get("reason") or "")
    if bool(payload.get("busy", False)) or reason in (nonfatal_reasons or set()):
        return "busy"
    if payload.get("ok") is False:
        return "error"
    return "ok"


def _step_record(result: dict[str, Any], *, nonfatal_reasons: set[str] | None = None) -> dict[str, Any]:
    return {
        "status": _step_status(result, nonfatal_reasons=nonfatal_reasons),
        "rc": int(result.get("rc", 1)),
        "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
        "timed_out": bool(result.get("timed_out", False)),
        "cmd": list(result.get("cmd") or []),
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def _prioritized_shards_for_core_focus(core_focus: dict[str, Any] | None) -> list[str]:
    shards = list(DEFAULT_DRAIN_SHARDS)
    if not isinstance(core_focus, dict) or not bool(core_focus.get("concentrated", False)):
        return shards

    top_rows = list(core_focus.get("hotspots") or [])
    top3_pending_lines = max(_safe_int(core_focus.get("top3_pending_lines"), 0), 0)
    material_pending_floor = max(1000, int(top3_pending_lines * 0.03))

    def _pending_for(predicate) -> int:
        pending = 0
        for row in top_rows[:5]:
            if not isinstance(row, dict):
                continue
            source = str(row.get("source_rel") or "").strip()
            if source and predicate(source):
                pending += max(_safe_int(row.get("pending_lines"), 0), 0)
        return pending

    governance_pending = _pending_for(
        lambda source: (
            source.startswith("governance/execution_lanes/")
            or source.startswith("governance/shadow_")
            or source.startswith("governance/events/")
            or source.startswith("governance/alerts/")
            or source.startswith("governance/distillation/")
        )
    )
    risk_support_pending = _pending_for(lambda source: source.startswith("governance/channels/risk/"))
    aggressive_pending = _pending_for(
        lambda source: (
            source.startswith("decisions/shadow_aggressive_")
            or source.startswith("decisions/shadow_intraday_aggressive_")
            or source.startswith("decisions/shadow_swing_aggressive_")
        )
    )
    crypto_pending = _pending_for(_is_crypto_decision_source)
    trading_pending = _pending_for(
        lambda source: (
            source.startswith("governance/channels/decision/")
            or (source.startswith("decisions/") and not (
                source.startswith("decisions/shadow_crypto/")
                or source.startswith("decisions/shadow_crypto_futures_crypto/")
                or "default_crypto_coinbase" in source
                or "crypto_futures_crypto_coinbase" in source
                or "default_crypto_schwab" in source
                or "crypto_futures_crypto_schwab" in source
            ))
        )
    )
    runtime_pending = _pending_for(lambda source: source.startswith("governance/channels/runtime/"))
    top3_share = _safe_float(core_focus.get("top3_share"), 0.0)
    severe_focus = bool(top3_share >= 0.9)
    support_floor = min(material_pending_floor, 25_000)

    if severe_focus:
        focus_only: list[str] = []
        for shard, pending in sorted(
            [
                ("governance", governance_pending),
                ("crypto_trading", crypto_pending),
                ("trading", trading_pending),
                ("aggressive_trading", aggressive_pending),
                ("risk_support", risk_support_pending),
                ("runtime", runtime_pending),
            ],
            key=lambda item: item[1],
            reverse=True,
        ):
            floor = support_floor if shard == "risk_support" else material_pending_floor
            if pending >= floor:
                focus_only.append(shard)
        focus_only.extend(["health_fast", "support_watchdog"])
        ordered_focus: list[str] = []
        for shard in focus_only:
            if shard in DEFAULT_DRAIN_SHARDS and shard not in ordered_focus:
                ordered_focus.append(shard)
        return ordered_focus or ["trading", "health_fast", "support_watchdog"]

    priority: list[str] = []
    if governance_pending >= material_pending_floor:
        priority.append("governance")
    if crypto_pending >= material_pending_floor:
        priority.append("crypto_trading")
    if aggressive_pending >= material_pending_floor:
        priority.append("aggressive_trading")
    if risk_support_pending >= support_floor:
        priority.append("risk_support")
    if trading_pending >= material_pending_floor:
        priority.append("trading")
    if runtime_pending >= material_pending_floor:
        priority.append("runtime")
    priority.extend(["health_fast", "trading_fast", "support_watchdog"])

    ordered: list[str] = []
    for shard in priority + shards:
        if shard not in ordered:
            ordered.append(shard)
    return ordered


def _massive_focus_active(core_focus: dict[str, Any] | None) -> bool:
    if not isinstance(core_focus, dict):
        return False
    rows = [row for row in list(core_focus.get("hotspots") or []) if isinstance(row, dict)]
    top3_pending_lines = max(_safe_int(core_focus.get("top3_pending_lines"), 0), 0)
    max_file_pending = max((_safe_int(row.get("pending_lines"), 0) for row in rows), default=0)
    return bool(
        bool(core_focus.get("concentrated", False))
        or top3_pending_lines >= MASSIVE_FOCUS_MIN_TOP3_LINES
        or max_file_pending >= MASSIVE_FOCUS_MIN_FILE_PENDING_LINES
        or any(_row_sparse_focus(row) for row in rows)
    )


def _row_sparse_focus(row: dict[str, Any]) -> bool:
    return bool(
        bool(row.get("sparse_large_line", False))
        and _safe_int(row.get("estimated_pending_bytes"), 0) >= SPARSE_LARGE_JSONL_PENDING_BYTES_FLOOR
    )


def _governance_focus_paths(core_focus: dict[str, Any] | None) -> list[str]:
    if not isinstance(core_focus, dict) or not _massive_focus_active(core_focus):
        return []

    hotspots = list(core_focus.get("hotspots") or [])
    focus_paths: list[str] = []
    for row in hotspots[:5]:
        if not isinstance(row, dict):
            continue
        source_rel = str(row.get("source_rel") or "").strip()
        if not (
            source_rel.startswith("governance/execution_lanes/")
            or source_rel.startswith("governance/shadow_")
            or source_rel.startswith("governance/events/")
            or source_rel.startswith("governance/alerts/")
            or source_rel.startswith("governance/distillation/")
        ):
            continue
        pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
        age_seconds = max(_safe_float(row.get("age_seconds"), 0.0), 0.0)
        if not _row_sparse_focus(row) and pending_lines < 25000 and age_seconds < 15 * 60:
            continue
        if source_rel.startswith("governance/events/") and pending_lines < 5000:
            continue
        if source_rel not in focus_paths:
            focus_paths.append(source_rel)
    return focus_paths


def _risk_support_focus_paths(core_focus: dict[str, Any] | None) -> list[str]:
    if not isinstance(core_focus, dict) or not _massive_focus_active(core_focus):
        return []

    focus_paths: list[str] = []
    for row in list(core_focus.get("hotspots") or [])[:8]:
        if not isinstance(row, dict):
            continue
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel.startswith("governance/channels/risk/"):
            continue
        pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
        age_seconds = max(_safe_float(row.get("age_seconds"), 0.0), 0.0)
        if not _row_sparse_focus(row) and pending_lines < 5000 and age_seconds < 15 * 60:
            continue
        if source_rel not in focus_paths:
            focus_paths.append(source_rel)
    return focus_paths


def _trading_focus_paths(core_focus: dict[str, Any] | None) -> list[str]:
    if not isinstance(core_focus, dict) or not _massive_focus_active(core_focus):
        return []

    hotspots = list(core_focus.get("hotspots") or [])
    focus_paths: list[str] = []
    for row in hotspots[:5]:
        if not isinstance(row, dict):
            continue
        source_rel = str(row.get("source_rel") or "").strip()
        if not (
            source_rel.startswith("governance/channels/decision/")
            or source_rel.startswith("decisions/")
        ):
            continue
        if _is_crypto_decision_source(source_rel) or _is_aggressive_decision_source(source_rel):
            continue
        pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
        age_seconds = max(_safe_float(row.get("age_seconds"), 0.0), 0.0)
        if pending_lines < 25000 and age_seconds < 15 * 60:
            continue
        if source_rel not in focus_paths:
            focus_paths.append(source_rel)
    return focus_paths


def _crypto_trading_focus_paths(core_focus: dict[str, Any] | None) -> list[str]:
    if not isinstance(core_focus, dict) or not _massive_focus_active(core_focus):
        return []

    focus_paths: list[str] = []
    for row in list(core_focus.get("hotspots") or [])[:5]:
        if not isinstance(row, dict):
            continue
        source_rel = str(row.get("source_rel") or "").strip()
        if not _is_crypto_decision_source(source_rel):
            continue
        pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
        age_seconds = max(_safe_float(row.get("age_seconds"), 0.0), 0.0)
        if not _row_sparse_focus(row) and pending_lines < 5000 and age_seconds < 15 * 60:
            continue
        if source_rel not in focus_paths:
            focus_paths.append(source_rel)
    return focus_paths


def _aggressive_trading_focus_paths(core_focus: dict[str, Any] | None) -> list[str]:
    if not isinstance(core_focus, dict) or not _massive_focus_active(core_focus):
        return []

    focus_paths: list[str] = []
    for row in list(core_focus.get("hotspots") or [])[:5]:
        if not isinstance(row, dict):
            continue
        source_rel = str(row.get("source_rel") or "").strip()
        if not _is_aggressive_decision_source(source_rel):
            continue
        pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
        age_seconds = max(_safe_float(row.get("age_seconds"), 0.0), 0.0)
        if pending_lines < 5000 and age_seconds < 15 * 60:
            continue
        if source_rel not in focus_paths:
            focus_paths.append(source_rel)
    return focus_paths


def _sparse_focus_shards(rows: list[dict[str, Any]]) -> set[str]:
    shards: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        shard = str(row.get("shard") or "").strip()
        if shard:
            shards.add(shard)
            continue
        source_rel = str(row.get("source_rel") or "").strip()
        if _is_crypto_decision_source(source_rel):
            shards.add("crypto_trading")
        elif _is_aggressive_decision_source(source_rel):
            shards.add("aggressive_trading")
        elif source_rel.startswith("governance/channels/decision/") or source_rel.startswith("decisions/"):
            shards.add("trading")
    return shards or {"trading", "aggressive_trading", "crypto_trading"}


def _is_crypto_decision_source(source_rel: str) -> bool:
    rel = str(source_rel or "")
    if not (rel.startswith("decisions/") or rel.startswith("governance/channels/decision/")):
        return False
    return any(
        part in rel
        for part in (
            "shadow_crypto/",
            "shadow_crypto_futures_crypto/",
            "default_crypto_coinbase",
            "crypto_futures_crypto_coinbase",
            "default_crypto_schwab",
            "crypto_futures_crypto_schwab",
        )
    )


def _is_crypto_explanation_source(source_rel: str) -> bool:
    rel = str(source_rel or "")
    if rel == "crypto_explanations" or "crypto_explanations" in rel:
        return True
    return any(
        part in rel
        for part in (
            "decision_explanations/shadow_crypto/",
            "decision_explanations/shadow_crypto_futures_crypto/",
            "decision_explanations/default_crypto_coinbase",
            "decision_explanations/crypto_futures_crypto_coinbase",
            "decision_explanations/default_crypto_schwab",
            "decision_explanations/crypto_futures_crypto_schwab",
        )
    )


def _is_aggressive_decision_source(source_rel: str) -> bool:
    rel = str(source_rel or "")
    return any(
        part in rel
        for part in (
            "shadow_aggressive_",
            "shadow_intraday_aggressive_",
            "shadow_swing_aggressive_",
        )
    )


def _deferred_focus_paths(
    backpressure: dict[str, Any] | None,
    *,
    predicate,
    row_predicate=None,
    exclude_row_predicate=None,
    min_pending_lines: int = 5_000,
    min_age_seconds: float = 15 * 60,
) -> list[str]:
    if not isinstance(backpressure, dict):
        return []
    focus_paths: list[str] = []
    for key in ("top_deferred_pending_files", "top_cold_pending_files", "top_pending_files", "_storage_overlay_sources"):
        rows = backpressure.get(key) if isinstance(backpressure.get(key), list) else []
        for row in rows[:8]:
            if not isinstance(row, dict):
                continue
            source_rel = str(row.get("source_rel") or "").strip()
            if not source_rel:
                continue
            if key == "_storage_overlay_sources" and exclude_row_predicate is not None and bool(exclude_row_predicate(row)):
                continue
            row_match = bool(row_predicate(row)) if row_predicate is not None and key == "_storage_overlay_sources" else False
            if not (predicate(source_rel) or row_match):
                continue
            pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
            age_seconds = max(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 0.0)
            if pending_lines < min_pending_lines and age_seconds < min_age_seconds:
                continue
            if source_rel not in focus_paths:
                focus_paths.append(source_rel)
    return focus_paths


def _explanation_focus_paths(backpressure: dict[str, Any] | None) -> list[str]:
    return _deferred_focus_paths(
        backpressure,
        predicate=lambda source: source.startswith("decision_explanations/") and not _is_crypto_explanation_source(source),
        exclude_row_predicate=lambda row: str(row.get("shard") or "").strip() == "crypto_explanations",
    )


def _crypto_explanation_focus_paths(backpressure: dict[str, Any] | None) -> list[str]:
    return _deferred_focus_paths(
        backpressure,
        predicate=_is_crypto_explanation_source,
        row_predicate=lambda row: str(row.get("shard") or "").strip() == "crypto_explanations",
    )


def _insert_priority_shards(base_shards: list[str], priority_shards: list[str]) -> list[str]:
    ordered: list[str] = []
    for shard in priority_shards + base_shards:
        if shard and shard in DEFAULT_DRAIN_SHARDS and shard not in ordered:
            ordered.append(shard)
    return ordered


def _storage_overlay_rows(storage_control: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(storage_control, dict):
        return []
    locator = storage_control.get("stale_pending_locator")
    if not isinstance(locator, dict):
        return []

    rows_by_source: dict[str, dict[str, Any]] = {}
    for key in ("top_pending_sources", "oldest_sources"):
        raw_rows = locator.get(key) if isinstance(locator.get(key), list) else []
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            source_rel = str(raw.get("source_rel") or "").strip()
            pending_lines = max(_safe_int(raw.get("pending_lines"), 0), 0)
            if not source_rel or pending_lines <= 0:
                continue
            age_seconds = max(_safe_float(raw.get("oldest_pending_age_seconds"), 0.0), 0.0)
            row = {
                "source_rel": source_rel,
                "pending_lines": pending_lines,
                "oldest_pending_age_seconds": age_seconds,
                "shard": str(raw.get("shard") or ""),
                "pressure_lane": str(raw.get("pressure_lane") or ""),
                "total_lines": _safe_int(raw.get("total_lines"), 0),
                "last_line": _safe_int(raw.get("last_line"), 0),
            }
            current = rows_by_source.get(source_rel)
            if current is None:
                rows_by_source[source_rel] = row
                continue
            current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending_lines)
            current["oldest_pending_age_seconds"] = max(
                _safe_float(current.get("oldest_pending_age_seconds"), 0.0),
                age_seconds,
            )
            if not str(current.get("shard") or ""):
                current["shard"] = row["shard"]
            if not str(current.get("pressure_lane") or ""):
                current["pressure_lane"] = row["pressure_lane"]

    rows = list(rows_by_source.values())
    rows.sort(
        key=lambda row: (
            _safe_int(row.get("pending_lines"), 0),
            _safe_float(row.get("oldest_pending_age_seconds"), 0.0),
        ),
        reverse=True,
    )
    return rows[:12]


def _merge_pending_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_source: dict[str, dict[str, Any]] = {}
    for rows in groups:
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            source_rel = str(raw.get("source_rel") or "").strip()
            pending_lines = max(_safe_int(raw.get("pending_lines"), 0), 0)
            if not source_rel or pending_lines <= 0:
                continue
            age_seconds = max(
                _safe_float(raw.get("oldest_pending_age_seconds"), _safe_float(raw.get("age_seconds"), 0.0)),
                0.0,
            )
            row = dict(raw)
            row["source_rel"] = source_rel
            row["pending_lines"] = pending_lines
            row["oldest_pending_age_seconds"] = age_seconds
            current = rows_by_source.get(source_rel)
            if current is None:
                rows_by_source[source_rel] = row
                continue
            if pending_lines > _safe_int(current.get("pending_lines"), 0):
                current.update(row)
            else:
                current["oldest_pending_age_seconds"] = max(
                    _safe_float(current.get("oldest_pending_age_seconds"), 0.0),
                    age_seconds,
                )
                if not str(current.get("shard") or ""):
                    current["shard"] = str(row.get("shard") or "")
                if not str(current.get("pressure_lane") or ""):
                    current["pressure_lane"] = str(row.get("pressure_lane") or "")

    merged = list(rows_by_source.values())
    merged.sort(
        key=lambda row: (
            _safe_int(row.get("pending_lines"), 0),
            _safe_float(row.get("oldest_pending_age_seconds"), 0.0),
        ),
        reverse=True,
    )
    return merged[:20]


def _backpressure_with_storage_overlay(
    backpressure: dict[str, Any] | None,
    storage_control: dict[str, Any] | None,
) -> dict[str, Any]:
    base = dict(backpressure or {})
    overlay_rows = _storage_overlay_rows(storage_control)
    if not overlay_rows:
        return base

    overlay_backpressure = {}
    if isinstance(storage_control, dict) and isinstance(storage_control.get("backpressure"), dict):
        overlay_backpressure = storage_control["backpressure"]

    top_pending_files = base.get("top_pending_files") if isinstance(base.get("top_pending_files"), list) else []
    base["top_pending_files"] = _merge_pending_rows(overlay_rows, top_pending_files)
    base["_storage_overlay_sources"] = overlay_rows
    base["_storage_overlay_adjusted"] = bool(overlay_backpressure.get("overlay_adjusted", False))

    overlay_core = _safe_int(overlay_backpressure.get("core_pending_lines"), 0)
    overlay_total = _safe_int(overlay_backpressure.get("total_pending_lines"), 0)
    overlay_deferred = _safe_int(overlay_backpressure.get("deferred_pending_lines"), 0)
    overlay_cold = _safe_int(overlay_backpressure.get("cold_pending_lines"), 0)
    overlay_oldest_age = _safe_float(overlay_backpressure.get("oldest_pending_age_seconds"), 0.0)
    if overlay_core > _safe_int(base.get("pending_lines"), 0):
        base["pending_lines"] = overlay_core
    if overlay_total > _safe_int(base.get("pending_lines_total"), 0):
        base["pending_lines_total"] = overlay_total
    if overlay_deferred > _safe_int(base.get("pending_lines_deferred"), 0):
        base["pending_lines_deferred"] = overlay_deferred
    if overlay_cold > _safe_int(base.get("pending_lines_cold"), 0):
        base["pending_lines_cold"] = overlay_cold
    if overlay_oldest_age > _safe_float(base.get("oldest_pending_age_seconds"), 0.0):
        base["oldest_pending_age_seconds"] = overlay_oldest_age
    return base


def _drop_empty_shard_path_filters(env: dict[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in env.items():
        key_text = str(key)
        is_shard_path_filter = (
            key_text.startswith("SQL_LINK_SERVICE_SHARD_")
            and (key_text.endswith("_PATH_CONTAINS") or key_text.endswith("_PATH_NOT_CONTAINS"))
        )
        if is_shard_path_filter and not str(value).strip():
            continue
        cleaned[key_text] = str(value)
    return cleaned


def _drain_env(
    base_env: dict[str, str],
    *,
    critical: bool,
    off_hours_active: bool,
    core_focus: dict[str, Any] | None = None,
    backpressure: dict[str, Any] | None = None,
) -> tuple[str, dict[str, str]]:
    env = {str(key): str(value) for key, value in base_env.items() if str(key).strip()}
    if not off_hours_active:
        return "standard_guard", env

    prioritized_shards = _prioritized_shards_for_core_focus(core_focus)
    governance_focus_paths = _governance_focus_paths(core_focus)
    risk_support_focus_paths = _risk_support_focus_paths(core_focus)
    trading_focus_paths = _trading_focus_paths(core_focus)
    crypto_trading_focus_paths = _crypto_trading_focus_paths(core_focus)
    aggressive_trading_focus_paths = _aggressive_trading_focus_paths(core_focus)
    explanation_focus_paths = _explanation_focus_paths(backpressure)
    crypto_explanation_focus_paths = _crypto_explanation_focus_paths(backpressure)
    core_hotspot_rows = list((core_focus or {}).get("hotspots") or []) if isinstance(core_focus, dict) else []
    sparse_focus_rows = [row for row in core_hotspot_rows if isinstance(row, dict) and _row_sparse_focus(row)]
    sparse_focus_active = bool(sparse_focus_rows)
    prioritized_shards = _insert_priority_shards(
        prioritized_shards,
        [
            "crypto_explanations" if crypto_explanation_focus_paths else "",
            "explanations" if explanation_focus_paths else "",
        ],
    )
    governance_first = bool(prioritized_shards and prioritized_shards[0] == "governance")
    governance_max_files = "14" if governance_first and critical else ("10" if governance_first else ("8" if critical else "6"))
    governance_max_lines = "64000" if governance_focus_paths else (
        "24000" if governance_first and critical else ("16000" if governance_first else ("4000" if critical else "6000"))
    )
    trading_focused = bool(trading_focus_paths)
    trading_first = bool(prioritized_shards and prioritized_shards[0] == "trading")
    trading_max_files = "16" if trading_focused and critical else ("14" if critical or trading_first else "12")
    trading_max_lines = "64000" if trading_focused else ("20000" if critical or trading_first else "14000")
    crypto_trading_focused = bool(crypto_trading_focus_paths)
    crypto_trading_first = bool(prioritized_shards and prioritized_shards[0] == "crypto_trading")
    crypto_trading_max_files = "16" if crypto_trading_focused and critical else ("14" if critical or crypto_trading_first else "10")
    crypto_trading_max_lines = "64000" if crypto_trading_focused else ("20000" if critical or crypto_trading_first else "14000")
    risk_support_focused = bool(risk_support_focus_paths)
    risk_support_first = bool(prioritized_shards and prioritized_shards[0] == "risk_support")
    risk_support_max_files = "6" if risk_support_focused and critical else ("6" if critical or risk_support_first else "4")
    risk_support_max_lines = "160000" if risk_support_focused and critical else ("120000" if critical or risk_support_first else "80000")
    risk_support_checkpoint_lines = "8000" if risk_support_focused and critical else ("6000" if critical or risk_support_first else "4000")
    aggressive_trading_focused = bool(aggressive_trading_focus_paths)
    aggressive_trading_first = bool(prioritized_shards and prioritized_shards[0] == "aggressive_trading")
    aggressive_trading_max_files = "14" if aggressive_trading_focused and critical else ("12" if critical or aggressive_trading_first else "10")
    aggressive_trading_max_lines = "32000" if aggressive_trading_focused else ("20000" if critical or aggressive_trading_first else "14000")
    explanations_focused = bool(explanation_focus_paths)
    crypto_explanations_focused = bool(crypto_explanation_focus_paths)
    explanations_max_files = "12" if explanations_focused and critical else ("8" if critical else "6")
    crypto_explanations_max_files = "14" if crypto_explanations_focused and critical else ("8" if critical else "6")
    explanations_max_lines = "64000" if explanations_focused else ("24000" if critical else "16000")
    crypto_explanations_max_lines = "64000" if crypto_explanations_focused else ("24000" if critical else "16000")

    env.update(
        {
            "INGEST_MAX_DEFERRED_FILES": "6" if critical else "4",
            "JSONL_SQL_MAX_COLD_LANE_FILES": "2" if critical else "1",
            "LOG_DATA_INGRESS": "0",
            "LOG_API_CALLS": "0",
            "LOG_LOOP_STATE": "0",
            "LOG_SHADOW_PNL_ATTRIBUTION": "0",
            "INGEST_JOURNAL_DAILY_ENABLED": "0",
            "INGEST_JOURNAL_FILE_START_ENABLED": "0",
            "INGEST_JOURNAL_CHECKPOINT_ENABLED": "0" if critical else "1",
            "INGEST_JOURNAL_ZERO_PENDING_ENABLED": "0",
            "INGEST_JOURNAL_ERRORS_ALWAYS": "1",
            "RESOURCE_GUARD_OPTIONAL_MAX_LOAD_PER_CORE": "12.0" if (critical or off_hours_active) else "4.0",
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12" if critical else "15",
            "SQL_LINK_SERVICE_IGNORE_ACTIVE_REQUEST": "1",
            "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "420" if critical else "240",
            "SQL_LINK_SERVICE_SHARDS": ",".join(prioritized_shards),
            "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "90" if critical else "60",
            "SQL_LINK_SERVICE_AUTO_HOT_RETENTION": "0",
            "SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION": "0",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "30",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000" if critical else "200000",
            "SQL_LINK_SERVICE_HOT_MAX_ROWS": "2400000" if critical else "1800000",
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "0.25" if critical else "0.5",
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "0.25" if critical else "0.5",
            "SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_FILES": "14" if critical else "12",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_MAX_FILES": "10" if critical else "8",
            "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES": governance_max_files,
            "SQL_LINK_SERVICE_SHARD_TRADING_MAX_FILES": trading_max_files,
            "SQL_LINK_SERVICE_SHARD_RUNTIME_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_GOVERNANCE_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_LINES_PER_FILE": "16000",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_MAX_LINES_PER_FILE": "16000",
            "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_LINES_PER_FILE": governance_max_lines,
            "SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS": ",".join(governance_focus_paths),
            "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS": ",".join(aggressive_trading_focus_paths),
            "SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS": ",".join(trading_focus_paths),
            "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS": ",".join(crypto_trading_focus_paths),
            "SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_PATH_CONTAINS": ",".join(risk_support_focus_paths),
            "SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_MAX_FILES": risk_support_max_files,
            "SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_MAX_LINES_PER_FILE": risk_support_max_lines,
            "SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_STATE_CHECKPOINT_LINES": risk_support_checkpoint_lines,
            "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_LINES_PER_FILE": "96000" if critical else "64000",
            "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_STATE_CHECKPOINT_LINES": "4000",
            "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_FILES": aggressive_trading_max_files,
            "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE": aggressive_trading_max_lines,
            "SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE": trading_max_lines,
            "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_FILES": crypto_trading_max_files,
            "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_LINES_PER_FILE": crypto_trading_max_lines,
            "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES": explanations_max_files,
            "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES": crypto_explanations_max_files,
            "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_LINES_PER_FILE": explanations_max_lines,
            "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_LINES_PER_FILE": crypto_explanations_max_lines,
            "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_PATH_CONTAINS": ",".join(explanation_focus_paths),
            "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_PATH_CONTAINS": ",".join(crypto_explanation_focus_paths),
            "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_MAX_FILES": "3" if critical else "2",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_MAX_FILES": "3" if critical else "2",
            "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "240000" if critical else "180000",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "220000" if critical else "160000",
            "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "260000" if critical else "220000",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "260000" if critical else "220000",
        }
    )
    if sparse_focus_active:
        sparse_shards = _sparse_focus_shards(sparse_focus_rows)
        env.update(
            {
                "SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_DRAIN": "1",
                "SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_FILE_COUNT": str(len(sparse_focus_rows)),
                "INGEST_MAX_BYTES_PER_FILE": str(SPARSE_LARGE_DECISION_MAX_BYTES_PER_FILE),
                "SQLITE_BATCH_MAX_BYTES": str(SPARSE_LARGE_DECISION_SQLITE_BATCH_MAX_BYTES),
                "SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_SHARDS": ",".join(sorted(sparse_shards)),
            }
        )
        if "trading" in sparse_shards:
            env.update(
                {
                    "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "250",
                    "SQL_LINK_SERVICE_SHARD_TRADING_MERGE_MAX_JSONL_ROWS": "250",
                }
            )
        if "aggressive_trading" in sparse_shards:
            env.update(
                {
                    "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "500",
                    "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MERGE_MAX_JSONL_ROWS": "250",
                }
            )
        if "crypto_trading" in sparse_shards:
            env.update(
                {
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_STATE_CHECKPOINT_LINES": "500",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MERGE_MAX_JSONL_ROWS": "250",
                }
            )
    return "offhours_external_backlog_drain", _drop_empty_shard_path_filters(env)


def _write_service_request(
    *,
    path: Path,
    drain_profile: str,
    drain_env: dict[str, str],
    wait_timeout_seconds: float,
    now_utc: datetime,
) -> dict[str, Any]:
    preserved = _preservable_service_request(path, now_utc=now_utc)
    if preserved:
        return preserved

    expires_utc = now_utc.timestamp() + max(float(wait_timeout_seconds), 900.0)
    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "active": True,
        "request_kind": "external_backlog_drain",
        "reason": str(drain_profile or "external_backlog_drain"),
        "requested_at": now_utc.isoformat(),
        "expires_utc": datetime.fromtimestamp(expires_utc, tz=timezone.utc).isoformat(),
        "env_overrides": {str(key): str(value) for key, value in drain_env.items() if str(key).strip()},
    }
    _write_json(path, payload)
    return payload


def _backpressure_snapshot(payload: dict[str, Any]) -> dict[str, int]:
    line_estimation = payload.get("line_estimation") if isinstance(payload.get("line_estimation"), dict) else {}
    return {
        "core_pending_lines": _safe_int(payload.get("pending_lines"), 0),
        "deferred_pending_lines": _safe_int(payload.get("pending_lines_deferred"), 0),
        "cold_pending_lines": _safe_int(payload.get("pending_lines_cold"), 0),
        "total_pending_lines": _safe_int(payload.get("pending_lines_total"), 0),
        "sparse_large_line_pending_lines": _safe_int(line_estimation.get("sparse_large_line_pending_lines"), 0),
        "sparse_large_line_pending_bytes": _safe_int(line_estimation.get("sparse_large_line_pending_bytes"), 0),
    }


def _hotspots(backpressure: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_source: dict[str, dict[str, Any]] = {}
    for lane, key in (
        ("deferred", "top_deferred_pending_files"),
        ("support", "top_support_telemetry_pending_files"),
        ("cold", "top_cold_pending_files"),
    ):
        raw_rows = backpressure.get(key)
        if not isinstance(raw_rows, list):
            continue
        for raw in raw_rows[:8]:
            if not isinstance(raw, dict):
                continue
            source_rel = str(raw.get("source_rel") or "").strip()
            if not source_rel:
                continue
            pending_lines = max(_safe_int(raw.get("pending_lines"), 0), 0)
            age_seconds = max(_safe_float(raw.get("oldest_pending_age_seconds"), 0.0), 0.0)
            if pending_lines <= 0:
                continue
            candidate_action = "drain_now"
            if source_rel.startswith("data/stale_stage/"):
                candidate_action = "reap_or_archive_stale_stage"
            elif lane == "support":
                candidate_action = "drain_support_watchdog"
            elif lane == "cold" and (age_seconds >= 6 * 60 * 60 or pending_lines >= 100000):
                candidate_action = "consider_archive_after_drain"
            elif lane == "deferred" and age_seconds >= 2 * 60 * 60:
                candidate_action = "drain_then_compact"
            current = rows_by_source.get(source_rel)
            row = {
                "lane": lane,
                "source_rel": source_rel,
                "pending_lines": pending_lines,
                "age_seconds": round(age_seconds, 3),
                "age_bucket": _age_bucket(age_seconds),
                "candidate_action": candidate_action,
            }
            if current is None:
                rows_by_source[source_rel] = row
                continue
            merged_lane = sorted({str(current.get("lane") or ""), lane})
            current["lane"] = ",".join(part for part in merged_lane if part)
            if pending_lines > int(current.get("pending_lines", 0) or 0):
                current["pending_lines"] = pending_lines
            if age_seconds > float(current.get("age_seconds", 0.0) or 0.0):
                current["age_seconds"] = round(age_seconds, 3)
                current["age_bucket"] = _age_bucket(age_seconds)
            preferred_actions = {
                "reap_or_archive_stale_stage": 3,
                "drain_support_watchdog": 2,
                "consider_archive_after_drain": 2,
                "drain_then_compact": 1,
                "drain_now": 0,
            }
            if preferred_actions.get(candidate_action, 0) > preferred_actions.get(str(current.get("candidate_action") or ""), 0):
                current["candidate_action"] = candidate_action
    rows = list(rows_by_source.values())
    rows.sort(
        key=lambda row: (
            int(row.get("pending_lines", 0)),
            float(row.get("age_seconds", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return rows[:10]


def _core_hotspots(backpressure: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_pending_lines = max(
        _safe_int(backpressure.get("pending_lines_total"), 0),
        _safe_int(backpressure.get("pending_lines"), 0),
        0,
    )
    top_pending_files = backpressure.get("top_pending_files") if isinstance(backpressure.get("top_pending_files"), list) else []
    for raw in top_pending_files[:5]:
        if not isinstance(raw, dict):
            continue
        source_rel = str(raw.get("source_rel") or "").strip()
        pending_lines = max(_safe_int(raw.get("pending_lines"), 0), 0)
        if not source_rel or pending_lines <= 0:
            continue
        rows.append(
            {
                "source_rel": source_rel,
                "pending_lines": pending_lines,
                "age_seconds": round(max(_safe_float(raw.get("oldest_pending_age_seconds"), 0.0), 0.0), 3),
                "sparse_large_line": bool(raw.get("sparse_large_line", False)),
                "estimated_pending_bytes": _safe_int(raw.get("estimated_pending_bytes"), 0),
                "estimated_avg_bytes_per_line": round(_safe_float(raw.get("estimated_avg_bytes_per_line"), 0.0), 3),
                "file_size_bytes": _safe_int(raw.get("file_size_bytes"), 0),
            }
        )
    top3_pending_lines = sum(int(row.get("pending_lines", 0) or 0) for row in rows[:3])
    return {
        "hotspots": rows,
        "top3_pending_lines": top3_pending_lines,
        "top3_share": round((top3_pending_lines / max(total_pending_lines, 1)) if total_pending_lines > 0 else 0.0, 6),
        "concentrated": bool(
            (top3_pending_lines / max(total_pending_lines, 1)) >= CORE_FOCUS_MIN_SHARE
            and (
                total_pending_lines >= CORE_FOCUS_MIN_PENDING_LINES
                or top3_pending_lines >= CORE_FOCUS_MIN_TOP3_LINES
            )
        ),
    }


def _material_drain_recommended(
    snapshot: dict[str, int],
    *,
    core_focus: dict[str, Any],
    aged_candidate_count: int,
    stale_stage_candidate_count: int,
    support_watchdog_pending_lines: int,
) -> bool:
    return bool(
        bool(core_focus.get("concentrated", False))
        or int(snapshot.get("total_pending_lines", 0) or 0) >= DRAIN_RECOMMEND_MIN_TOTAL_PENDING_LINES
        or int(snapshot.get("deferred_pending_lines", 0) or 0) >= DRAIN_RECOMMEND_MIN_DEFERRED_PENDING_LINES
        or int(snapshot.get("cold_pending_lines", 0) or 0) >= DRAIN_RECOMMEND_MIN_DEFERRED_PENDING_LINES
        or int(support_watchdog_pending_lines) >= DRAIN_RECOMMEND_MIN_SUPPORT_PENDING_LINES
        or int(snapshot.get("sparse_large_line_pending_bytes", 0) or 0) >= SPARSE_LARGE_JSONL_PENDING_BYTES_FLOOR
        or int(aged_candidate_count) > 0
        or int(stale_stage_candidate_count) > 0
    )


def _follow_through_progress_signature(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_step": str(payload.get("current_step") or ""),
        "completed_shard_count": _safe_int(payload.get("completed_shard_count"), 0),
        "completed_merge_count": _safe_int(payload.get("completed_merge_count"), 0),
        "merged_rows_this_cycle": _safe_int(payload.get("merged_rows_this_cycle"), 0),
    }


def _follow_through_progressed(previous: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    numeric_keys = ("completed_shard_count", "completed_merge_count", "merged_rows_this_cycle")
    if previous is None:
        return any(int(current.get(key, 0) or 0) > 0 for key in numeric_keys)
    if any(int(current.get(key, 0) or 0) > int(previous.get(key, 0) or 0) for key in numeric_keys):
        return True
    previous_step = str(previous.get("current_step") or "")
    current_step = str(current.get("current_step") or "")
    return bool(previous_step and current_step and current_step != previous_step)


def _follow_through_retry(
    *,
    project_root: Path,
    health_root: Path,
    drain_env: dict[str, str],
    poll_seconds: float,
    wait_timeout_seconds: float,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    deadline = started.timestamp() + max(float(wait_timeout_seconds), 0.0)
    attempts = 0
    last_result: dict[str, Any] | None = None
    observed_writer_pid = _lock_owner_pid(SQL_WRITER_LOCK_PATH)
    previous_signature: dict[str, Any] | None = None
    last_progress_signature: dict[str, Any] = {}
    progress_events = 0

    while datetime.now(timezone.utc).timestamp() <= deadline:
        attempts += 1
        result = _run_json_command(
            [str(PY), str(project_root / "scripts" / "ops" / "sql_link_shard_manager.py"), "--once", "--json"],
            cwd=project_root,
            payload_path=health_root / "sql_link_service_latest.json",
            env_overrides=drain_env,
        )
        last_result = result
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        signature = _follow_through_progress_signature(payload)
        if signature and _follow_through_progressed(previous_signature, signature):
            progress_events += 1
            last_progress_signature = signature
        if signature:
            previous_signature = signature
        status = _step_status(result, nonfatal_reasons={"writer_lock_busy"})
        if status != "busy":
            break
        observed_writer_pid = observed_writer_pid or _lock_owner_pid(SQL_WRITER_LOCK_PATH)
        sleep_seconds = max(float(poll_seconds), 0.1)
        remaining = max(deadline - datetime.now(timezone.utc).timestamp(), 0.0)
        if remaining <= 0.0:
            break
        time_mod.sleep(min(sleep_seconds, remaining))

    waited_seconds = max((datetime.now(timezone.utc) - started).total_seconds(), 0.0)
    final_status = _step_status(last_result or {}, nonfatal_reasons={"writer_lock_busy"})
    completed = bool(last_result is not None and final_status != "busy")
    progress_observed = progress_events > 0
    return {
        "requested": True,
        "completed": completed,
        "timed_out": not completed,
        "attempts": attempts,
        "poll_seconds": round(float(poll_seconds), 3),
        "wait_timeout_seconds": round(float(wait_timeout_seconds), 3),
        "waited_seconds": round(waited_seconds, 3),
        "observed_writer_pid": observed_writer_pid,
        "status": "completed" if completed else "timed_out",
        "progress_observed": progress_observed,
        "progress_events": progress_events,
        "progress_state": "completed" if completed else ("progressing" if progress_observed else "stalled"),
        "last_progress_signature": last_progress_signature,
        "last_result": last_result or {},
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    force_live_window: bool = False,
    resource_profile: str = "optional",
    follow_through: bool = False,
    poll_seconds: float = 20.0,
    wait_timeout_seconds: float = 900.0,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"
    backpressure_before = _load_json(health_root / "ingestion_backpressure_latest.json")
    storage_control_before = _load_json(health_root / "ingestion_storage_control_latest.json")
    queue_before = _load_json(health_root / "ingestion_priority_queue_latest.json")
    governor_payload = governor_src.build_payload(
        project_root,
        override_path=project_root / "config" / ".env.storage_pressure_override",
        action="status",
        changed=False,
    )
    window = _off_hours_window(now)
    mount = _load_json(health_root / "storage_mount_guard_latest.json")
    split_brain = _load_json(health_root / "storage_split_brain_reconciler_latest.json")
    unresolved_conflicts = _safe_int(((split_brain.get("summary") or {}).get("unresolved_conflicts")), 0)
    external_available = bool(mount.get("external_available", False))
    storage_mode = str(mount.get("storage_mode") or "")
    critical = str(governor_payload.get("profile") or "") == "critical_backpressure"
    planning_backpressure_before = _backpressure_with_storage_overlay(backpressure_before, storage_control_before)
    core_focus_before = _core_hotspots(planning_backpressure_before)
    drain_profile, drain_env = _drain_env(
        governor_payload.get("env_overrides") if isinstance(governor_payload.get("env_overrides"), dict) else {},
        critical=critical,
        off_hours_active=bool(window.get("active", False) or force_live_window),
        core_focus=core_focus_before,
        backpressure=planning_backpressure_before,
    )

    blocked_reasons: list[str] = []
    if not external_available or storage_mode != "external":
        blocked_reasons.append("external_storage_unavailable")
    if unresolved_conflicts > 0:
        blocked_reasons.append("split_brain_unresolved")
    if not bool(window.get("active", False)) and not force_live_window:
        blocked_reasons.append("market_hours_guard")

    apply_executed = False
    backpressure_after = backpressure_before
    storage_control_after = storage_control_before
    queue_after = queue_before
    steps: dict[str, Any] = {}
    writer_busy = False
    service_request_payload: dict[str, Any] = {}
    follow_through_summary = {
        "requested": bool(follow_through),
        "completed": False,
        "timed_out": False,
        "attempts": 0,
        "poll_seconds": round(float(poll_seconds), 3),
        "wait_timeout_seconds": round(float(wait_timeout_seconds), 3),
        "waited_seconds": 0.0,
        "observed_writer_pid": _lock_owner_pid(SQL_WRITER_LOCK_PATH),
        "progress_observed": False,
        "progress_events": 0,
        "progress_state": "not_requested" if not follow_through else "not_needed",
        "last_progress_signature": {},
        "status": "not_requested" if not follow_through else "not_needed",
    }

    if apply:
        steps["ingestion_backpressure_before"] = _step_record(
            _run_json_command(
                [str(PY), str(project_root / "scripts" / "ingestion_backpressure_guard.py"), "--json"],
                cwd=project_root,
                payload_path=health_root / "ingestion_backpressure_latest.json",
            )
        )
        if steps["ingestion_backpressure_before"]["status"] != "error":
            refreshed = _load_json(health_root / "ingestion_backpressure_latest.json")
            if refreshed:
                backpressure_before = refreshed
                storage_control_before = _load_json(health_root / "ingestion_storage_control_latest.json") or storage_control_before
                refreshed_planning_backpressure = _backpressure_with_storage_overlay(backpressure_before, storage_control_before)
                refreshed_core_focus = _core_hotspots(refreshed_planning_backpressure)
                if refreshed_core_focus.get("hotspots") or refreshed_planning_backpressure.get("_storage_overlay_sources"):
                    planning_backpressure_before = refreshed_planning_backpressure
                    core_focus_before = refreshed_core_focus
                    drain_profile, drain_env = _drain_env(
                        governor_payload.get("env_overrides") if isinstance(governor_payload.get("env_overrides"), dict) else {},
                        critical=critical,
                        off_hours_active=bool(window.get("active", False) or force_live_window),
                        core_focus=core_focus_before,
                        backpressure=planning_backpressure_before,
                    )

        steps["ingestion_priority_queue_before"] = _step_record(
            _run_json_command(
                [str(PY), str(project_root / "scripts" / "ops" / "ingestion_priority_queue.py"), "--json"],
                cwd=project_root,
                payload_path=health_root / "ingestion_priority_queue_latest.json",
            )
        )
        if steps["ingestion_priority_queue_before"]["status"] != "error":
            refreshed = _load_json(health_root / "ingestion_priority_queue_latest.json")
            if refreshed:
                queue_before = refreshed

        if not blocked_reasons:
            resource_guard = _run_json_command(
                [str(PY), str(project_root / "scripts" / "resource_guard.py"), "--profile", str(resource_profile or "optional"), "--json"],
                cwd=project_root,
                env_overrides=drain_env,
            )
            steps["resource_guard"] = _step_record(resource_guard)
            resource_payload = resource_guard.get("payload") if isinstance(resource_guard.get("payload"), dict) else {}
            resource_ok = bool(resource_payload.get("ok", resource_payload.get("resource_guard_ok", False)))
            if resource_ok:
                apply_executed = True
                shard_manager_initial = _run_json_command(
                    [str(PY), str(project_root / "scripts" / "ops" / "sql_link_shard_manager.py"), "--once", "--json"],
                    cwd=project_root,
                    payload_path=health_root / "sql_link_service_latest.json",
                    env_overrides=drain_env,
                )
                writer_busy = _step_status(shard_manager_initial, nonfatal_reasons={"writer_lock_busy"}) == "busy"
                shard_manager = shard_manager_initial
                steps["sql_link_shard_manager_initial"] = _step_record(shard_manager_initial, nonfatal_reasons={"writer_lock_busy"})
                if writer_busy:
                    service_request_path = health_root / "sql_link_service_request_latest.json"
                    service_request_payload = _write_service_request(
                        path=service_request_path,
                        drain_profile=drain_profile,
                        drain_env=drain_env,
                        wait_timeout_seconds=wait_timeout_seconds,
                        now_utc=now,
                    )
                    steps["sql_link_service_request"] = {
                        "status": "ok",
                        "rc": 0,
                        "duration_ms": 0.0,
                        "timed_out": False,
                        "cmd": ["write", str(service_request_path)],
                        "stdout_tail": json.dumps(service_request_payload, ensure_ascii=True),
                        "stderr_tail": "",
                    }
                    if follow_through:
                        follow_through_summary = {
                            **follow_through_summary,
                            "completed": True,
                            "timed_out": False,
                            "attempts": 1,
                            "waited_seconds": 0.0,
                            "observed_writer_pid": _lock_owner_pid(SQL_WRITER_LOCK_PATH),
                            "progress_observed": False,
                            "progress_events": 0,
                            "progress_state": "requested_live_writer",
                            "last_progress_signature": {},
                            "status": "handoff_requested",
                        }
                steps["sql_link_shard_manager"] = _step_record(shard_manager, nonfatal_reasons={"writer_lock_busy"})
                sqlite_maintenance = _run_json_command(
                    [str(PY), str(project_root / "scripts" / "sqlite_performance_maintenance.py"), "--checkpoint-only", "--json"],
                    cwd=project_root,
                    payload_path=health_root / "sqlite_maintenance_latest.json",
                    env_overrides=drain_env,
                    timeout_seconds=20.0,
                )
                steps["sqlite_maintenance"] = _step_record(sqlite_maintenance, nonfatal_reasons={"timeout"})
                stale_sweeper = _run_json_command(
                    [str(PY), str(project_root / "scripts" / "ops" / "stale_artifact_sweeper_bot.py"), "--json"],
                    cwd=project_root,
                    payload_path=health_root / "stale_artifact_sweeper_bot_latest.json",
                    env_overrides=drain_env,
                )
                steps["stale_artifact_sweeper_bot"] = _step_record(stale_sweeper, nonfatal_reasons={"already_running", "lock_busy"})
                stale_reaper = _run_json_command(
                    [str(PY), str(project_root / "scripts" / "ops" / "stale_artifact_reaper_bot.py"), "--json"],
                    cwd=project_root,
                    payload_path=health_root / "stale_artifact_reaper_bot_latest.json",
                    env_overrides=drain_env,
                )
                steps["stale_artifact_reaper_bot"] = _step_record(stale_reaper, nonfatal_reasons={"already_running", "lock_busy"})
                retention = _run_json_command(
                    [
                        str(PY),
                        str(project_root / "scripts" / "data_retention_policy.py"),
                        "--apply",
                        "--no-stale-stage",
                        "--no-stale-purge",
                        "--json",
                    ],
                    cwd=project_root,
                    payload_path=health_root / "data_retention_latest.json",
                    env_overrides=drain_env,
                )
                steps["data_retention_policy"] = _step_record(retention, nonfatal_reasons={"lock_busy"})
            else:
                blocked_reasons.append("resource_guard_blocked")
                if bool(core_focus_before.get("concentrated", False)) and _lock_owner_pid(SQL_WRITER_LOCK_PATH):
                    service_request_path = health_root / "sql_link_service_request_latest.json"
                    service_request_payload = _write_service_request(
                        path=service_request_path,
                        drain_profile=f"{drain_profile}:resource_guard_handoff",
                        drain_env=drain_env,
                        wait_timeout_seconds=wait_timeout_seconds,
                        now_utc=now,
                    )
                    steps["sql_link_service_request"] = {
                        "status": "ok",
                        "rc": 0,
                        "duration_ms": 0.0,
                        "timed_out": False,
                        "cmd": ["write", str(service_request_path)],
                        "stdout_tail": json.dumps(service_request_payload, ensure_ascii=True),
                        "stderr_tail": "",
                    }
                    if follow_through:
                        follow_through_summary = {
                            **follow_through_summary,
                            "completed": True,
                            "timed_out": False,
                            "attempts": 1,
                            "waited_seconds": 0.0,
                            "observed_writer_pid": _lock_owner_pid(SQL_WRITER_LOCK_PATH),
                            "progress_observed": False,
                            "progress_events": 0,
                            "progress_state": "requested_live_writer_after_resource_guard",
                            "last_progress_signature": {},
                            "status": "handoff_requested",
                        }

        if apply_executed:
            steps["ingestion_backpressure_after"] = _step_record(
                _run_json_command(
                    [str(PY), str(project_root / "scripts" / "ingestion_backpressure_guard.py"), "--json"],
                    cwd=project_root,
                    payload_path=health_root / "ingestion_backpressure_latest.json",
                )
            )
            refreshed = _load_json(health_root / "ingestion_backpressure_latest.json")
            if refreshed:
                backpressure_after = refreshed
                storage_control_after = _load_json(health_root / "ingestion_storage_control_latest.json") or storage_control_before
            steps["ingestion_priority_queue_after"] = _step_record(
                _run_json_command(
                    [str(PY), str(project_root / "scripts" / "ops" / "ingestion_priority_queue.py"), "--json"],
                    cwd=project_root,
                    payload_path=health_root / "ingestion_priority_queue_latest.json",
                )
            )
            refreshed = _load_json(health_root / "ingestion_priority_queue_latest.json")
            if refreshed:
                queue_after = refreshed

    planning_backpressure_before = _backpressure_with_storage_overlay(backpressure_before, storage_control_before)
    planning_backpressure_after = _backpressure_with_storage_overlay(
        backpressure_after if apply_executed else backpressure_before,
        storage_control_after if apply_executed else storage_control_before,
    )
    before_snapshot = _backpressure_snapshot(planning_backpressure_before)
    after_snapshot = _backpressure_snapshot(planning_backpressure_after)
    raw_before_snapshot = _backpressure_snapshot(backpressure_before)
    raw_after_snapshot = _backpressure_snapshot(backpressure_after)
    hotspots = _hotspots(backpressure_after if apply_executed else backpressure_before)
    core_focus = _core_hotspots(planning_backpressure_after)
    aged_candidates = [
        row
        for row in hotspots
        if str(row.get("candidate_action") or "") in {
            "consider_archive_after_drain",
            "drain_then_compact",
            "reap_or_archive_stale_stage",
        }
    ]
    stale_stage_candidates = [
        row for row in hotspots if str(row.get("candidate_action") or "") == "reap_or_archive_stale_stage"
    ]
    support_watchdog_candidates = [
        row for row in hotspots if str(row.get("candidate_action") or "") == "drain_support_watchdog"
    ]
    top_actions: list[str] = []
    if "external_storage_unavailable" in blocked_reasons:
        top_actions.append("keep the writer on the routed local path until external BOT_LOGS storage is healthy again")
    if "split_brain_unresolved" in blocked_reasons:
        top_actions.append("resolve split-brain conflicts before draining the external backlog")
    if "market_hours_guard" in blocked_reasons:
        top_actions.append("wait for the off-hours window before raising deferred and cold drain quotas")
    if "resource_guard_blocked" in blocked_reasons:
        top_actions.append("rerun the external backlog drain after memory and disk guards return to green")
    if aged_candidates:
        top_actions.append("compact or archive the oldest deferred and cold backlog files after the active drain pass")
    if stale_stage_candidates:
        top_actions.append("reap or archive staged stale artifacts after the active drain pass so cold backlog stops recycling")
    if support_watchdog_candidates:
        top_actions.append("let the watchdog support shard drain failover and pager logs off the main governance path")
    if bool(core_focus.get("concentrated", False)):
        top_actions.append("keep the writer focused on the dominant core backlog files before widening deferred or cold drain budgets")
    governance_focus_paths = [part for part in str(drain_env.get("SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS") or "").split(",") if part]
    risk_support_focus_paths = [part for part in str(drain_env.get("SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_PATH_CONTAINS") or "").split(",") if part]
    trading_focus_paths = [part for part in str(drain_env.get("SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS") or "").split(",") if part]
    crypto_trading_focus_paths = [part for part in str(drain_env.get("SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS") or "").split(",") if part]
    aggressive_trading_focus_paths = [part for part in str(drain_env.get("SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS") or "").split(",") if part]
    explanation_focus_paths = [part for part in str(drain_env.get("SQL_LINK_SERVICE_SHARD_EXPLANATIONS_PATH_CONTAINS") or "").split(",") if part]
    crypto_explanation_focus_paths = [part for part in str(drain_env.get("SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_PATH_CONTAINS") or "").split(",") if part]
    if governance_focus_paths:
        top_actions.append("keep the governance shard pinned to the dominant governance backlog files until those queue anchors step down")
    if trading_focus_paths:
        top_actions.append("keep the trading shard pinned to the dominant decision-channel files until the core queue falls below the halt threshold")
    if crypto_trading_focus_paths:
        top_actions.append("keep the crypto trading shard pinned to stale crypto decision files until the crypto queue catches up")
    if risk_support_focus_paths:
        top_actions.append("keep the risk-support shard pinned to stale risk-channel files so support pressure cannot crowd core ingestion")
    if aggressive_trading_focus_paths:
        top_actions.append("keep the aggressive trading shard pinned to high-velocity aggressive decision files until the intraday queue catches up")
    if explanation_focus_paths:
        top_actions.append("keep the explanations shard pinned to stale decision-explanation files until deferred backlog steps down")
    if crypto_explanation_focus_paths:
        top_actions.append("keep the crypto explanations shard pinned to stale crypto explanation files until the deferred crypto backlog catches up")
    if follow_through and follow_through_summary["status"] == "timed_out":
        if str(follow_through_summary.get("progress_state") or "") == "progressing":
            top_actions.append("the automatic follow-through timed out, but the SQL writer was still advancing shard or merge work, so let the current maintenance window run or extend the timeout next pass")
        else:
            top_actions.append("the automatic follow-through timed out without any observed shard or merge progress, so rerun during a quieter maintenance window")
    if str(follow_through_summary.get("status") or "") == "handoff_requested":
        top_actions.append("the active SQL writer accepted a live drain request and will apply the backlog-drain overrides on its next cycle")
    if writer_busy:
        if str(follow_through_summary.get("progress_state") or "") == "progressing":
            top_actions.append("let the current SQL writer finish the active drain cycle before forcing another external backlog pass")
        elif str(follow_through_summary.get("status") or "") == "handoff_requested":
            top_actions.append("let the current SQL writer roll into the requested drain cycle before judging deferred or cold progress")
        else:
            top_actions.append("rerun the external backlog drain after the current SQL writer lock holder finishes")
    if after_snapshot["deferred_pending_lines"] > 0 or after_snapshot["cold_pending_lines"] > 0:
        top_actions.append("repeat the external backlog drain during off-hours until deferred and cold queues stay below target")
    if after_snapshot["total_pending_lines"] > 0:
        top_actions.append("keep shadow attribution and channel logging throttled while the external backlog burns down")

    material_drain = _material_drain_recommended(
        after_snapshot,
        core_focus=core_focus,
        aged_candidate_count=len(aged_candidates),
        stale_stage_candidate_count=len(stale_stage_candidates),
        support_watchdog_pending_lines=sum(int(row.get("pending_lines", 0) or 0) for row in support_watchdog_candidates),
    )
    recommended_now = bool(window.get("active", False) and not blocked_reasons and material_drain)
    hard_blocked_reasons = [reason for reason in blocked_reasons if reason != "market_hours_guard"]
    soft_blocked_reasons = [reason for reason in blocked_reasons if reason == "market_hours_guard"]
    waiting_for_off_hours = bool(
        not apply_executed
        and material_drain
        and not hard_blocked_reasons
        and soft_blocked_reasons == ["market_hours_guard"]
    )
    ok = not bool(hard_blocked_reasons)
    if apply_executed:
        overall_status = "drain_active"
    elif waiting_for_off_hours:
        overall_status = "waiting_for_off_hours"
    elif ok:
        overall_status = "ready"
    else:
        overall_status = "blocked"
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "apply_requested": bool(apply),
        "apply_executed": bool(apply_executed),
        "follow_through": follow_through_summary,
        "recommended_now": recommended_now,
        "material_drain_recommended": material_drain,
        "blocked_reasons": blocked_reasons,
        "hard_blocked_reasons": hard_blocked_reasons,
        "soft_blocked_reasons": soft_blocked_reasons,
        "waiting_for_off_hours": waiting_for_off_hours,
        "off_hours_window": window,
        "drain_profile": drain_profile,
        "governor_profile": str(governor_payload.get("profile") or ""),
        "storage_mode": storage_mode,
        "writer_busy": bool(writer_busy),
        "service_request_path": str(health_root / "sql_link_service_request_latest.json"),
        "service_request": service_request_payload,
        "backpressure_before": before_snapshot,
        "backpressure_after": after_snapshot,
        "raw_backpressure_before": raw_before_snapshot,
        "raw_backpressure_after": raw_after_snapshot,
        "storage_overlay_focus": {
            "active": bool((planning_backpressure_after if apply_executed else planning_backpressure_before).get("_storage_overlay_sources")),
            "adjusted": bool((planning_backpressure_after if apply_executed else planning_backpressure_before).get("_storage_overlay_adjusted")),
            "sources": list((planning_backpressure_after if apply_executed else planning_backpressure_before).get("_storage_overlay_sources") or [])[:8],
        },
        "drain_delta": {
            key: int(before_snapshot.get(key, 0) - after_snapshot.get(key, 0))
            for key in ("core_pending_lines", "deferred_pending_lines", "cold_pending_lines", "total_pending_lines")
        },
        "hotspots": hotspots,
        "core_hotspots": core_focus.get("hotspots") or [],
        "core_focus_top3_pending_lines": _safe_int(core_focus.get("top3_pending_lines"), 0),
        "core_focus_top3_share": _safe_float(core_focus.get("top3_share"), 0.0),
        "core_focus_concentrated": bool(core_focus.get("concentrated", False)),
        "aged_candidate_files": len(aged_candidates),
        "aged_candidate_pending_lines": sum(int(row.get("pending_lines", 0) or 0) for row in aged_candidates),
        "stale_stage_candidate_files": len(stale_stage_candidates),
        "stale_stage_candidate_pending_lines": sum(int(row.get("pending_lines", 0) or 0) for row in stale_stage_candidates),
        "support_watchdog_candidate_files": len(support_watchdog_candidates),
        "support_watchdog_candidate_pending_lines": sum(int(row.get("pending_lines", 0) or 0) for row in support_watchdog_candidates),
        "queue_depth_before": _safe_int(queue_before.get("queue_depth"), 0),
        "queue_depth_after": _safe_int(queue_after.get("queue_depth"), _safe_int(queue_before.get("queue_depth"), 0)),
        "drain_overrides": {
            "deferred_files_budget": _safe_int(drain_env.get("INGEST_MAX_DEFERRED_FILES"), 0),
            "cold_files_budget": _safe_int(drain_env.get("JSONL_SQL_MAX_COLD_LANE_FILES"), 0),
            "sql_interval_seconds": _safe_int(drain_env.get("SQL_LINK_SERVICE_INTERVAL_SECONDS"), 0),
            "auto_hot_retention": str(drain_env.get("SQL_LINK_SERVICE_AUTO_HOT_RETENTION") or ""),
            "auto_queue_retention": str(drain_env.get("SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION") or ""),
            "hot_batch_size": _safe_int(drain_env.get("SQL_LINK_SERVICE_HOT_BATCH_SIZE"), 0),
            "resource_guard_optional_max_load_per_core": _safe_float(
                drain_env.get("RESOURCE_GUARD_OPTIONAL_MAX_LOAD_PER_CORE"),
                0.0,
            ),
            "sparse_large_decision_drain": str(drain_env.get("SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_DRAIN") or "") == "1",
            "sparse_large_decision_file_count": _safe_int(drain_env.get("SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_FILE_COUNT"), 0),
            "ingest_max_bytes_per_file": _safe_int(drain_env.get("INGEST_MAX_BYTES_PER_FILE"), 0),
            "sqlite_batch_max_bytes": _safe_int(drain_env.get("SQLITE_BATCH_MAX_BYTES"), 0),
            "preferred_shards": [part for part in str(drain_env.get("SQL_LINK_SERVICE_SHARDS") or "").split(",") if part],
            "governance_max_files": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES"), 0),
            "governance_max_lines_per_file": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_LINES_PER_FILE"), 0),
            "governance_path_focus": governance_focus_paths,
            "trading_max_files": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_TRADING_MAX_FILES"), 0),
            "trading_max_lines_per_file": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE"), 0),
            "trading_path_focus": trading_focus_paths,
            "crypto_trading_max_files": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_FILES"), 0),
            "crypto_trading_max_lines_per_file": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_LINES_PER_FILE"), 0),
            "crypto_trading_path_focus": crypto_trading_focus_paths,
            "risk_support_max_files": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_MAX_FILES"), 0),
            "risk_support_max_lines_per_file": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_MAX_LINES_PER_FILE"), 0),
            "risk_support_path_focus": risk_support_focus_paths,
            "aggressive_trading_max_files": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_FILES"), 0),
            "aggressive_trading_max_lines_per_file": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"), 0),
            "aggressive_trading_path_focus": aggressive_trading_focus_paths,
            "explanations_max_files": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES"), 0),
            "explanations_max_lines_per_file": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_LINES_PER_FILE"), 0),
            "explanations_path_focus": explanation_focus_paths,
            "crypto_explanations_max_files": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES"), 0),
            "crypto_explanations_max_lines_per_file": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_LINES_PER_FILE"), 0),
            "crypto_explanations_path_focus": crypto_explanation_focus_paths,
            "shard_link_timeout_seconds": _safe_int(drain_env.get("SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"), 0),
        },
        "steps": steps,
        "top_actions": top_actions[:8],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an off-hours external backlog drain for deferred and cold ingestion lanes.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force-live-window", action="store_true")
    parser.add_argument("--resource-profile", default="optional")
    parser.add_argument("--follow-through", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        apply=bool(args.apply),
        force_live_window=bool(args.force_live_window),
        resource_profile=str(args.resource_profile or "optional"),
        follow_through=bool(args.follow_through),
        poll_seconds=float(args.poll_seconds),
        wait_timeout_seconds=float(args.wait_timeout_seconds),
    )
    out_path = Path(args.out_file).expanduser()
    _write_json(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "external_backlog_drain "
            f"status={payload.get('overall_status', '')} "
            f"recommended_now={int(bool(payload.get('recommended_now', False)))} "
            f"apply_executed={int(bool(payload.get('apply_executed', False)))}"
        )
    return 0 if bool(payload.get("ok", False) or payload.get("apply_executed", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
