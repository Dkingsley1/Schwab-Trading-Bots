#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import gzip
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import runtime_training_common as rtc
from scripts.ops.long_runtime_common import eastern_off_hours_window


DEFAULT_ROWS_PATH = PROJECT_ROOT / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
DEFAULT_HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_training_snapshot_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "runtime_training_snapshot.lock"
_FILE_HASH_CHUNK_BYTES = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_FILE_HASH_CHUNK_BYTES), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _acquire_single_flight_lock(
    lock_path: Path,
    *,
    project_root: Path,
    health_path: Path,
    rows_path: Path,
) -> tuple[Any | None, dict[str, Any]]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        existing = handle.read().strip()
        try:
            age_seconds = max(time.time() - lock_path.stat().st_mtime, 0.0)
        except Exception:
            age_seconds = 0.0
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": True,
            "overall_status": "already_running",
            "already_running": True,
            "project_root": str(project_root),
            "health_path": str(health_path),
            "rows_path": str(rows_path),
            "lock_path": str(lock_path),
            "lock_age_seconds": round(float(age_seconds), 3),
            "existing_lock": existing,
            "single_flight_contract": {
                "active": True,
                "prevents_duplicate_snapshot_builders": True,
                "policy": "return already_running instead of launching overlapping runtime-training snapshot scans",
            },
        }
        handle.close()
        return None, payload
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "argv": sys.argv,
                "project_root": str(project_root),
                "health_path": str(health_path),
                "rows_path": str(rows_path),
            },
            ensure_ascii=True,
        )
        + "\n"
    )
    handle.flush()
    return handle, {}


def _parse_ts(raw: str) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        ts = datetime.fromisoformat(text)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    except Exception:
        return None


def _summary_latest_row_timestamp(summary: dict[str, Any]) -> datetime | None:
    candidates = [
        summary.get("latest_row_timestamp_utc"),
        summary.get("source_max_timestamp_utc"),
    ]
    coverage = summary.get("coverage") if isinstance(summary.get("coverage"), dict) else {}
    candidates.append(coverage.get("latest_row_timestamp_utc"))
    for row in coverage.get("top_sequences") if isinstance(coverage.get("top_sequences"), list) else []:
        if isinstance(row, dict):
            candidates.append(row.get("last_timestamp_utc"))
    parsed = [ts for raw in candidates if (ts := _parse_ts(raw)) is not None]
    return max(parsed) if parsed else None


def _snapshot_content_freshness(
    summary: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    latest = _summary_latest_row_timestamp(summary)
    market_window = eastern_off_hours_window(now=current)
    max_age_minutes = (
        _env_int("RUNTIME_TRAIN_SNAPSHOT_OFF_HOURS_CONTENT_MAX_AGE_MINUTES", 4320)
        if bool(market_window.get("active", False))
        else _env_int("RUNTIME_TRAIN_SNAPSHOT_MARKET_HOURS_CONTENT_MAX_AGE_MINUTES", 180)
    )
    age_minutes = max((current - latest).total_seconds(), 0.0) / 60.0 if latest is not None else None
    return {
        "content_fresh": bool(age_minutes is not None and age_minutes <= max(max_age_minutes, 1)),
        "latest_row_timestamp_utc": latest.isoformat() if latest is not None else "",
        "content_age_minutes": round(float(age_minutes), 4) if age_minutes is not None else None,
        "content_max_age_minutes": max(max_age_minutes, 1),
        "market_window": market_window,
    }
def _reusable_snapshot_payload(
    summary: dict[str, Any],
    *,
    project_root: Path,
    lookback_days: int,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
    prefer_sqlite: bool,
    max_age_minutes: int,
) -> dict[str, Any]:
    if max(int(max_age_minutes), 0) <= 0:
        return {}
    if not isinstance(summary, dict) or not summary:
        return {}
    if str(summary.get("project_root") or "") != str(project_root):
        return {}
    if int(summary.get("lookback_days", 0) or 0) < max(int(lookback_days), 1):
        return {}
    if [str(x) for x in summary.get("mode_allowlist", [])] != list(mode_allowlist):
        return {}
    if [str(x) for x in summary.get("symbol_allowlist", [])] != list(symbol_allowlist):
        return {}
    if bool(summary.get("prefer_sqlite", False)) != bool(prefer_sqlite):
        return {}
    if int(summary.get("sequence_count", 0) or 0) <= 0 or int(summary.get("row_count", 0) or 0) <= 0:
        return {}
    rows_path = Path(str(summary.get("rows_path") or "")).expanduser()
    if not rows_path.exists():
        return {}
    ts = _parse_ts(summary.get("timestamp_utc"))
    if ts is None:
        return {}
    age_minutes = max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0) / 60.0
    if age_minutes > float(max(int(max_age_minutes), 0)):
        return {}
    content_freshness = _snapshot_content_freshness(summary)
    if not bool(content_freshness.get("content_fresh", False)):
        return {}
    payload = dict(summary)
    payload.update(content_freshness)
    payload["reused"] = True
    payload["reuse_reason"] = "fresh_compatible_snapshot"
    payload["age_minutes"] = round(float(age_minutes), 4)
    return payload


def _light_refresh_existing_snapshot_payload(
    summary: dict[str, Any],
    *,
    project_root: Path,
    health_path: Path,
    lookback_days: int,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
    prefer_sqlite: bool,
) -> dict[str, Any]:
    if not _summary_config_compatible(
        summary,
        project_root=project_root,
        lookback_days=lookback_days,
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
        prefer_sqlite=prefer_sqlite,
    ):
        return {}
    if int(summary.get("sequence_count", 0) or 0) <= 0 or int(summary.get("row_count", 0) or 0) <= 0:
        return {}
    rows_path = Path(str(summary.get("rows_path") or "")).expanduser()
    if not rows_path.exists():
        return {}
    content_freshness = _snapshot_content_freshness(summary)
    if not bool(content_freshness.get("content_fresh", False)):
        return {}
    payload = dict(summary)
    payload.update(content_freshness)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    payload["health_path"] = str(health_path)
    payload["reused"] = True
    payload["reuse_reason"] = "light_refresh_existing_snapshot"
    payload["build_mode"] = "light_metadata_refresh"
    payload["age_minutes"] = 0.0
    return payload


def _summary_config_compatible(
    summary: dict[str, Any],
    *,
    project_root: Path,
    lookback_days: int,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
    prefer_sqlite: bool,
) -> bool:
    if not isinstance(summary, dict) or not summary:
        return False
    if str(summary.get("project_root") or "") != str(project_root):
        return False
    if int(summary.get("lookback_days", 0) or 0) < max(int(lookback_days), 1):
        return False
    if [str(x) for x in summary.get("mode_allowlist", [])] != list(mode_allowlist):
        return False
    if [str(x) for x in summary.get("symbol_allowlist", [])] != list(symbol_allowlist):
        return False
    if bool(summary.get("prefer_sqlite", False)) != bool(prefer_sqlite):
        return False
    rows_path = Path(str(summary.get("rows_path") or "")).expanduser()
    return rows_path.exists()


def _summary_can_seed_target(
    summary: dict[str, Any],
    *,
    project_root: Path,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
) -> bool:
    if not isinstance(summary, dict) or not summary:
        return False
    if str(summary.get("project_root") or "") != str(project_root):
        return False
    rows_path = Path(str(summary.get("rows_path") or "")).expanduser()
    if not rows_path.exists():
        return False
    seed_mode_allowlist = [str(x) for x in summary.get("mode_allowlist", [])]
    seed_symbol_allowlist = [str(x) for x in summary.get("symbol_allowlist", [])]
    if seed_mode_allowlist and any(str(mode) not in seed_mode_allowlist for mode in mode_allowlist):
        return False
    if seed_symbol_allowlist and any(str(symbol) not in seed_symbol_allowlist for symbol in symbol_allowlist):
        return False
    return int(summary.get("lookback_days", 0) or 0) > 0


def _path_mtime_utc(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _logical_runtime_source_key(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    return f"{path.parent.resolve()}::{name}"


def _prefer_runtime_source(current: Path | None, candidate: Path) -> Path:
    if current is None:
        return candidate
    if current.suffix == ".gz" and candidate.suffix != ".gz":
        return candidate
    if current.suffix != ".gz" and candidate.suffix == ".gz":
        return current
    current_mtime = _path_mtime_utc(current) or datetime.fromtimestamp(0, tz=timezone.utc)
    candidate_mtime = _path_mtime_utc(candidate) or datetime.fromtimestamp(0, tz=timezone.utc)
    if candidate_mtime > current_mtime:
        return candidate
    if candidate_mtime < current_mtime:
        return current
    try:
        candidate_size = candidate.stat().st_size
    except Exception:
        candidate_size = -1
    try:
        current_size = current.stat().st_size
    except Exception:
        current_size = -1
    if candidate_size > current_size:
        return candidate
    if candidate_size < current_size:
        return current
    return current


def _incremental_candidate_paths(
    project_root: Path,
    *,
    lookback_days: int,
    since_utc: datetime,
) -> list[Path]:
    return _window_candidate_paths(
        project_root,
        lookback_days=lookback_days,
        since_utc=since_utc,
        before_utc=None,
    )


def _window_candidate_paths(
    project_root: Path,
    *,
    lookback_days: int,
    since_utc: datetime,
    before_utc: datetime | None,
) -> list[Path]:
    chosen: dict[str, Path] = {}
    for path in rtc._recent_decision_paths(project_root, lookback_days=max(int(lookback_days), 1)):
        day_utc = rtc._path_day_utc(path)
        mtime_utc = _path_mtime_utc(path)
        if day_utc is None and mtime_utc is None:
            continue
        include = False
        if day_utc is not None:
            include = day_utc >= since_utc
            if not include and day_utc.date() >= since_utc.date():
                include = True
            if include and before_utc is not None:
                include = day_utc < before_utc and day_utc.date() < before_utc.date()
        if not include and mtime_utc is not None:
            include = mtime_utc >= since_utc and (before_utc is None or mtime_utc < before_utc)
        if not include:
            continue
        key = _logical_runtime_source_key(path)
        chosen[key] = _prefer_runtime_source(chosen.get(key), path)
    return sorted(chosen.values())


def _path_mtime_sort_key(path: Path) -> tuple[float, str]:
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    return (float(mtime), str(path))


def _iter_json_rows(
    paths: Iterable[Path],
    *,
    max_rows: int = 0,
    deadline_monotonic: float | None = None,
    stats: dict[str, Any] | None = None,
) -> Iterable[dict[str, Any]]:
    parsed_rows = 0
    for path in paths:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            if stats is not None:
                stats["timed_out"] = True
            break
        try:
            if path.suffix == ".gz":
                handle_cm = gzip.open(path, "rt", encoding="utf-8")
            else:
                handle_cm = path.open("r", encoding="utf-8")
            with handle_cm as handle:
                for line in handle:
                    if max_rows > 0 and parsed_rows >= max_rows:
                        if stats is not None:
                            stats["row_limit_hit"] = True
                        return
                    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                        if stats is not None:
                            stats["timed_out"] = True
                        return
                    if stats is not None:
                        stats["candidate_line_count"] = int(stats.get("candidate_line_count", 0) or 0) + 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        parsed_rows += 1
                        if stats is not None:
                            stats["candidate_json_row_count"] = int(stats.get("candidate_json_row_count", 0) or 0) + 1
                        yield row
        except Exception:
            if stats is not None:
                stats["candidate_file_error_count"] = int(stats.get("candidate_file_error_count", 0) or 0) + 1
            continue


def _iter_recent_json_rows_newest_first(
    paths: Iterable[Path],
    *,
    since_utc: datetime,
    max_rows: int = 0,
    deadline_monotonic: float | None = None,
    stats: dict[str, Any] | None = None,
    block_bytes: int = 256 * 1024,
) -> Iterable[dict[str, Any]]:
    parsed_rows = 0
    for path in paths:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            if stats is not None:
                stats["timed_out"] = True
            return
        if path.suffix == ".gz":
            for row in _iter_json_rows(
                [path],
                max_rows=max(max_rows - parsed_rows, 0) if max_rows else 0,
                deadline_monotonic=deadline_monotonic,
                stats=stats,
            ):
                timestamp = _parse_ts(row.get("timestamp_utc"))
                if timestamp is not None and timestamp >= since_utc:
                    parsed_rows += 1
                    yield row
            continue
        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                position = handle.tell()
                pending = b""
                seen_recent = False
                while position > 0:
                    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                        if stats is not None:
                            stats["timed_out"] = True
                        return
                    size = min(max(int(block_bytes), 1024), position)
                    position -= size
                    handle.seek(position)
                    pending = handle.read(size) + pending
                    lines = pending.splitlines()
                    if position > 0:
                        pending = lines[0] if lines else pending
                        complete_lines = lines[1:]
                    else:
                        pending = b""
                        complete_lines = lines
                    for raw_line in reversed(complete_lines):
                        if max_rows > 0 and parsed_rows >= max_rows:
                            if stats is not None:
                                stats["row_limit_hit"] = True
                            return
                        if stats is not None:
                            stats["candidate_line_count"] = int(stats.get("candidate_line_count", 0) or 0) + 1
                        try:
                            row = json.loads(raw_line)
                        except Exception:
                            continue
                        if not isinstance(row, dict):
                            continue
                        timestamp = _parse_ts(row.get("timestamp_utc"))
                        if timestamp is None:
                            continue
                        if timestamp < since_utc:
                            if seen_recent:
                                break
                            continue
                        seen_recent = True
                        parsed_rows += 1
                        if stats is not None:
                            stats["candidate_json_row_count"] = int(stats.get("candidate_json_row_count", 0) or 0) + 1
                        yield row
                    else:
                        continue
                    break
        except Exception:
            if stats is not None:
                stats["candidate_file_error_count"] = int(stats.get("candidate_file_error_count", 0) or 0) + 1


def _normalize_runtime_observation(
    row: dict[str, Any],
    *,
    since_utc: datetime,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
    price_sidecar: dict[str, Any] | None = None,
) -> tuple[tuple[str, str], dict[str, Any]] | None:
    metadata = rtc._runtime_row_metadata(row)
    strategy = rtc._runtime_row_strategy(row, metadata)
    strategy_priority = rtc._runtime_strategy_priority(strategy, metadata)
    if strategy_priority is None:
        return None
    ts = _parse_ts(row.get("timestamp_utc"))
    if ts is None or ts < since_utc:
        return None
    mode = rtc._runtime_row_mode(row, metadata)
    symbol = str(row.get("symbol") or "").strip().upper()
    if not mode or not symbol:
        return None
    if mode_allowlist and mode not in {str(x).strip().lower() for x in mode_allowlist}:
        return None
    if symbol_allowlist and symbol not in {str(x).strip().upper() for x in symbol_allowlist}:
        return None

    gates = row.get("gates") if isinstance(row.get("gates"), dict) else {}
    if ("market_data_ok" in gates) and (not bool(gates.get("market_data_ok"))):
        return None

    snapshot_ids = rtc._runtime_snapshot_id_candidates(row, metadata)
    features = rtc._runtime_row_features(row)
    price = rtc._runtime_row_price(row, features)
    if price_sidecar:
        sidecar_context = rtc._lookup_runtime_sidecar_context(
            price_sidecar,
            symbol=symbol,
            snapshot_ids=snapshot_ids,
            ts=ts,
        )
        sidecar_price = rtc._runtime_sidecar_entry_price(sidecar_context)
        if price <= 0.0 and sidecar_price > 0.0:
            price = sidecar_price
        if sidecar_context:
            features = rtc._runtime_features_with_sidecar_context(features, sidecar_context)
    if price <= 0.0:
        return None

    snapshot_id = snapshot_ids[0] if snapshot_ids else ""
    if not snapshot_id:
        snapshot_id = f"{symbol}:{ts.isoformat()}"

    obs = {
        "timestamp_utc": ts.isoformat(),
        "strategy": strategy,
        "strategy_priority": int(strategy_priority),
        "snapshot_id": snapshot_id,
        "ts_epoch": float(ts.timestamp()),
        "price": price,
        "features": features,
        "mode": mode,
        "symbol": symbol,
    }
    return (mode, symbol), obs


def _carry_forward_features(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    features = rows[-1].get("features") if isinstance(rows[-1].get("features"), dict) else {}
    carry: dict[str, float] = {}
    for key, value in features.items():
        try:
            numeric = float(value)
        except Exception:
            continue
        if numeric == numeric:
            carry[str(key)] = numeric
    return carry


def _merge_candidate_rows_into_sequences(
    base_sequences: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    candidate_paths: list[Path],
    project_root: Path,
    since_utc: datetime,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
    max_runtime_seconds: float = 0.0,
    max_candidate_rows: int = 0,
) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    deadline = started + max_runtime_seconds if max_runtime_seconds > 0 else None
    candidate_paths = sorted(candidate_paths, key=_path_mtime_sort_key, reverse=True)
    scan_stats: dict[str, Any] = {
        "candidate_scan_budget_seconds": round(float(max_runtime_seconds), 3),
        "candidate_scan_max_rows": int(max_candidate_rows),
        "candidate_source_count": len(candidate_paths),
        "candidate_scan_timed_out": False,
        "candidate_scan_row_limit_hit": False,
        "candidate_line_count": 0,
        "candidate_json_row_count": 0,
        "candidate_file_error_count": 0,
    }
    price_sidecar: dict[str, Any] = {}
    if candidate_paths and rtc._env_flag("RUNTIME_TRAIN_PRICE_SIDECAR_ENABLED", True):
        sidecar_max_rows = max(rtc._safe_int(os.getenv("RUNTIME_TRAIN_PRICE_SIDECAR_MAX_ROWS"), 5000), 0)
        price_sidecar = rtc._build_runtime_price_sidecar_from_rows(
            rtc._iter_runtime_price_sidecar_rows(candidate_paths, max_rows=sidecar_max_rows),
            max_rows=sidecar_max_rows,
        )

    best_by_snapshot: dict[tuple[str, str, str], dict[str, Any]] = {}
    row_iter_stats: dict[str, Any] = {
        "candidate_line_count": 0,
        "candidate_json_row_count": 0,
        "candidate_file_error_count": 0,
        "source_quota_hit_count": 0,
        "timed_out": False,
        "row_limit_hit": False,
    }
    global_row_budget = max(int(max_candidate_rows), 0)
    per_source_row_budget = (
        max((global_row_budget + len(candidate_paths) - 1) // max(len(candidate_paths), 1), 1)
        if global_row_budget > 0
        else 0
    )
    for path in candidate_paths:
        consumed_rows = int(row_iter_stats["candidate_json_row_count"])
        if global_row_budget > 0 and consumed_rows >= global_row_budget:
            row_iter_stats["row_limit_hit"] = True
            break
        if deadline is not None and time.monotonic() >= deadline:
            row_iter_stats["timed_out"] = True
            break
        remaining_budget = max(global_row_budget - consumed_rows, 0) if global_row_budget > 0 else 0
        path_budget = min(per_source_row_budget, remaining_budget) if global_row_budget > 0 else 0
        path_stats: dict[str, Any] = {}
        for row in _iter_recent_json_rows_newest_first(
            [path],
            since_utc=since_utc,
            max_rows=path_budget,
            deadline_monotonic=deadline,
            stats=path_stats,
        ):
            normalized = _normalize_runtime_observation(
                row,
                since_utc=since_utc,
                mode_allowlist=mode_allowlist,
                symbol_allowlist=symbol_allowlist,
                price_sidecar=price_sidecar,
            )
            if normalized is None:
                continue
            (mode, symbol), obs = normalized
            key = (mode, symbol, str(obs.get("snapshot_id") or ""))
            prev = best_by_snapshot.get(key)
            if prev is None or int(obs.get("strategy_priority", 99)) < int(prev.get("strategy_priority", 99)):
                best_by_snapshot[key] = obs
        row_iter_stats["candidate_line_count"] += int(path_stats.get("candidate_line_count", 0) or 0)
        row_iter_stats["candidate_json_row_count"] += int(path_stats.get("candidate_json_row_count", 0) or 0)
        row_iter_stats["candidate_file_error_count"] += int(path_stats.get("candidate_file_error_count", 0) or 0)
        if bool(path_stats.get("row_limit_hit", False)):
            row_iter_stats["source_quota_hit_count"] += 1
        if bool(path_stats.get("timed_out", False)):
            row_iter_stats["timed_out"] = True
            break
    if global_row_budget > 0 and int(row_iter_stats["candidate_json_row_count"]) >= global_row_budget:
        row_iter_stats["row_limit_hit"] = True
    scan_stats.update(
        {
            "candidate_scan_timed_out": bool(row_iter_stats.get("timed_out", False)),
            "candidate_scan_row_limit_hit": bool(row_iter_stats.get("row_limit_hit", False)),
            "candidate_line_count": int(row_iter_stats.get("candidate_line_count", 0) or 0),
            "candidate_json_row_count": int(row_iter_stats.get("candidate_json_row_count", 0) or 0),
            "candidate_file_error_count": int(row_iter_stats.get("candidate_file_error_count", 0) or 0),
            "candidate_per_source_row_budget": int(per_source_row_budget),
            "candidate_source_quota_hit_count": int(row_iter_stats.get("source_quota_hit_count", 0) or 0),
            "candidate_scan_fair_share": True,
        }
    )

    gap_fill_context = rtc._load_runtime_gap_fill_context(project_root)
    merged_row_count = 0
    grouped_new: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (mode, symbol, _snapshot_id), obs in best_by_snapshot.items():
        grouped_new[(mode, symbol)].append(obs)
    for key, new_rows in grouped_new.items():
        existing = list(base_sequences.get(key) or [])
        seen_snapshot_ids = {str(row.get("snapshot_id") or "") for row in existing}
        carry_forward = _carry_forward_features(existing)
        for row in sorted(
            new_rows,
            key=lambda item: (
                float(item.get("ts_epoch", 0.0)),
                int(item.get("strategy_priority", 99)),
                str(item.get("snapshot_id") or ""),
            ),
        ):
            snapshot_id = str(row.get("snapshot_id") or "")
            if snapshot_id in seen_snapshot_ids:
                continue
            enriched = rtc._enrich_runtime_observation(
                row,
                carry_forward_features=carry_forward,
                gap_fill_context=gap_fill_context,
            )
            existing.append(enriched)
            carry_forward = _carry_forward_features(existing)
            seen_snapshot_ids.add(snapshot_id)
            merged_row_count += 1
        if existing:
            existing.sort(
                key=lambda item: (
                    float(item.get("ts_epoch", 0.0)),
                    int(item.get("strategy_priority", 99)),
                    str(item.get("snapshot_id") or ""),
                )
            )
            base_sequences[key] = existing
    scan_stats["candidate_scan_elapsed_seconds"] = round(float(time.monotonic() - started), 3)
    scan_stats["candidate_scan_partial"] = bool(
        scan_stats["candidate_scan_timed_out"] or scan_stats["candidate_scan_row_limit_hit"]
    )
    return int(merged_row_count), scan_stats


def _incremental_snapshot_sequences(
    summary: dict[str, Any],
    *,
    project_root: Path,
    health_path: Path,
    lookback_days: int,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
    prefer_sqlite: bool,
    max_runtime_seconds: float = 0.0,
    max_candidate_rows: int = 0,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]] | None:
    if not _summary_config_compatible(
        summary,
        project_root=project_root,
        lookback_days=lookback_days,
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
        prefer_sqlite=prefer_sqlite,
    ):
        return None
    since_summary_utc = _summary_latest_row_timestamp(summary) or _parse_ts(summary.get("timestamp_utc"))
    if since_summary_utc is None:
        return None

    base_sequences = rtc._load_runtime_snapshot_rows(
        project_root,
        lookback_days=max(int(lookback_days), 1),
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
        snapshot_file=health_path,
    )
    if not base_sequences:
        return None

    candidate_paths = _incremental_candidate_paths(
        project_root,
        lookback_days=max(int(lookback_days), 1),
        since_utc=since_summary_utc,
    )
    if not candidate_paths:
        return base_sequences, {
            "build_mode": "incremental_refresh",
            "incremental_base_timestamp_utc": since_summary_utc.isoformat(),
            "incremental_source_count": 0,
            "incremental_source_paths": [],
            "incremental_row_count": 0,
        }

    since_utc = datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))
    merged_row_count, scan_stats = _merge_candidate_rows_into_sequences(
        base_sequences,
        candidate_paths=candidate_paths,
        project_root=project_root,
        since_utc=since_utc,
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
        max_runtime_seconds=max_runtime_seconds,
        max_candidate_rows=max_candidate_rows,
    )

    return base_sequences, {
        "build_mode": "incremental_refresh",
        "incremental_base_timestamp_utc": since_summary_utc.isoformat(),
        "incremental_source_count": len(candidate_paths),
        "incremental_source_paths": [str(path) for path in candidate_paths[:20]],
        "incremental_row_count": int(merged_row_count),
        "incremental_partial": bool(scan_stats.get("candidate_scan_partial", False)),
        "incremental_scan": scan_stats,
    }


def _full_refresh_sequences(
    project_root: Path,
    *,
    lookback_days: int,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
    prefer_sqlite: bool,
    max_observation_rows: int,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    sequences = rtc.load_runtime_observation_sequences(
        project_root,
        lookback_days=max(int(lookback_days), 1),
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
        prefer_sqlite=bool(prefer_sqlite),
        allow_snapshot=False,
        max_observation_rows=max(int(max_observation_rows), 0) or None,
    )
    meta: dict[str, Any] = {"build_mode": "full_refresh"}
    if sequences or not bool(prefer_sqlite):
        return sequences, meta

    fallback_sequences = rtc.load_runtime_observation_sequences(
        project_root,
        lookback_days=max(int(lookback_days), 1),
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
        prefer_sqlite=False,
        allow_snapshot=False,
        max_observation_rows=max(int(max_observation_rows), 0) or None,
    )
    if fallback_sequences:
        return fallback_sequences, {
            "build_mode": "full_refresh_jsonl_fallback",
            "sqlite_empty_fallback": True,
            "fallback_reason": "sqlite_preferred_snapshot_returned_zero_sequences",
        }
    meta["sqlite_empty_fallback"] = True
    meta["fallback_reason"] = "sqlite_and_jsonl_sources_returned_zero_sequences"
    return sequences, meta


def _seeded_snapshot_sequences(
    seed_summary: dict[str, Any],
    *,
    seed_health_path: Path,
    project_root: Path,
    lookback_days: int,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]] | None:
    if not _summary_can_seed_target(
        seed_summary,
        project_root=project_root,
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
    ):
        return None

    target_lookback_days = max(int(lookback_days), 1)
    seed_lookback_days = max(int(seed_summary.get("lookback_days", 0) or 0), 1)
    base_lookback_days = min(seed_lookback_days, target_lookback_days)
    base_sequences = rtc._load_runtime_snapshot_rows(
        project_root,
        lookback_days=base_lookback_days,
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
        snapshot_file=seed_health_path,
    )
    if not base_sequences:
        return None

    now_utc = datetime.now(timezone.utc)
    target_since_utc = now_utc - timedelta(days=target_lookback_days)
    seed_since_utc = now_utc - timedelta(days=base_lookback_days)
    seed_summary_utc = _parse_ts(seed_summary.get("timestamp_utc"))

    chosen: dict[str, Path] = {}
    if target_lookback_days > base_lookback_days:
        for path in _window_candidate_paths(
            project_root,
            lookback_days=target_lookback_days,
            since_utc=target_since_utc,
            before_utc=seed_since_utc,
        ):
            key = _logical_runtime_source_key(path)
            chosen[key] = _prefer_runtime_source(chosen.get(key), path)
    if seed_summary_utc is not None:
        for path in _window_candidate_paths(
            project_root,
            lookback_days=target_lookback_days,
            since_utc=seed_summary_utc,
            before_utc=None,
        ):
            key = _logical_runtime_source_key(path)
            chosen[key] = _prefer_runtime_source(chosen.get(key), path)

    candidate_paths = sorted(chosen.values())
    if not candidate_paths:
        return base_sequences, {
            "build_mode": "seed_backfill_refresh",
            "seed_health_path": str(seed_health_path),
            "seed_base_lookback_days": int(base_lookback_days),
            "seed_source_count": 0,
            "seed_source_paths": [],
            "seed_backfill_row_count": 0,
        }

    merged_row_count, scan_stats = _merge_candidate_rows_into_sequences(
        base_sequences,
        candidate_paths=candidate_paths,
        project_root=project_root,
        since_utc=target_since_utc,
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
    )
    return base_sequences, {
        "build_mode": "seed_backfill_refresh",
        "seed_health_path": str(seed_health_path),
        "seed_base_lookback_days": int(base_lookback_days),
        "seed_source_count": len(candidate_paths),
        "seed_source_paths": [str(path) for path in candidate_paths[:20]],
        "seed_backfill_row_count": int(merged_row_count),
        "seed_backfill_partial": bool(scan_stats.get("candidate_scan_partial", False)),
        "seed_backfill_scan": scan_stats,
    }


def _coverage_summary(
    sequences: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    recent_window_hours = (1, 2, 6, 24)
    recent: dict[int, dict[str, Any]] = {
        hours: {"row_count": 0, "snapshot_ids": set(), "symbols": set()}
        for hours in recent_window_hours
    }
    mode_row_counts: dict[str, int] = {}
    mode_sequence_counts: dict[str, int] = {}
    symbol_row_counts: dict[str, int] = {}
    sequence_rows: list[dict[str, Any]] = []
    parsed_timestamps: list[datetime] = []
    for (mode, symbol), rows in sorted(sequences.items()):
        row_count = int(len(rows))
        if row_count <= 0:
            continue
        mode_row_counts[mode] = int(mode_row_counts.get(mode, 0) + row_count)
        mode_sequence_counts[mode] = int(mode_sequence_counts.get(mode, 0) + 1)
        symbol_row_counts[symbol] = int(symbol_row_counts.get(symbol, 0) + row_count)
        for row in rows:
            timestamp = _parse_ts(row.get("timestamp_utc"))
            if timestamp is None:
                continue
            age_hours = max((current - timestamp).total_seconds(), 0.0) / 3600.0
            for hours, bucket in recent.items():
                if age_hours > float(hours):
                    continue
                bucket["row_count"] = int(bucket["row_count"]) + 1
                bucket["symbols"].add(str(row.get("symbol") or symbol).strip().upper())
                snapshot_id = str(row.get("snapshot_id") or "").strip()
                if snapshot_id:
                    bucket["snapshot_ids"].add(snapshot_id)
        first_ts = str(rows[0].get("timestamp_utc") or "") if rows else ""
        last_ts = str(rows[-1].get("timestamp_utc") or "") if rows else ""
        for raw in (first_ts, last_ts):
            parsed = _parse_ts(raw)
            if parsed is not None:
                parsed_timestamps.append(parsed)
        sequence_rows.append(
            {
                "mode": mode,
                "symbol": symbol,
                "row_count": row_count,
                "first_timestamp_utc": first_ts,
                "last_timestamp_utc": last_ts,
            }
        )
    sequence_rows.sort(key=lambda row: (-int(row["row_count"]), row["mode"], row["symbol"]))
    top_modes = [
        {
            "mode": mode,
            "sequence_count": int(mode_sequence_counts[mode]),
            "row_count": int(mode_row_counts[mode]),
        }
        for mode in sorted(mode_row_counts, key=lambda item: (-mode_row_counts[item], item))[:20]
    ]
    top_symbols = [
        {
            "symbol": symbol,
            "row_count": int(symbol_row_counts[symbol]),
        }
        for symbol in sorted(symbol_row_counts, key=lambda item: (-symbol_row_counts[item], item))[:20]
    ]
    return {
        "mode_count": int(len(mode_sequence_counts)),
        "symbol_count": int(len(symbol_row_counts)),
        "top_modes": top_modes,
        "top_symbols": top_symbols,
        "top_sequences": sequence_rows[:25],
        "earliest_row_timestamp_utc": min(parsed_timestamps).isoformat() if parsed_timestamps else "",
        "latest_row_timestamp_utc": max(parsed_timestamps).isoformat() if parsed_timestamps else "",
        "recent_windows": {
            str(hours): {
                "window_hours": hours,
                "window_ended_utc": current.isoformat(),
                "row_count": int(bucket["row_count"]),
                "rows_with_snapshot_id": len(bucket["snapshot_ids"]),
                "unique_snapshot_ids": len(bucket["snapshot_ids"]),
                "unique_symbols": len(bucket["symbols"]),
            }
            for hours, bucket in recent.items()
        },
    }


def _limit_snapshot_sequences(
    sequences: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    max_sequences: int,
    max_rows_per_sequence: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    sequence_cap = max(int(max_sequences), 0)
    row_cap = max(int(max_rows_per_sequence), 0)
    ranked = sorted(sequences.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1]))
    if sequence_cap:
        ranked = ranked[:sequence_cap]
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, rows in ranked:
        rows_sorted = sorted(
            list(rows),
            key=lambda item: (
                float(item.get("ts_epoch", 0.0)),
                int(item.get("strategy_priority", 99)),
                str(item.get("snapshot_id") or ""),
            ),
        )
        if row_cap and len(rows_sorted) > row_cap:
            rows_sorted = rows_sorted[-row_cap:]
        if rows_sorted:
            out[key] = rows_sorted
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a canonical runtime-training snapshot for future retrains.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--lookback-days", type=int, default=int(os.getenv("RUNTIME_TRAIN_SNAPSHOT_LOOKBACK_DAYS", "14")))
    parser.add_argument("--mode-allowlist", default=os.getenv("RUNTIME_TRAIN_SNAPSHOT_MODE_ALLOWLIST", ""))
    parser.add_argument("--symbol-allowlist", default=os.getenv("RUNTIME_TRAIN_SNAPSHOT_SYMBOL_ALLOWLIST", ""))
    parser.add_argument("--prefer-sqlite", action=argparse.BooleanOptionalAction, default=os.getenv("RUNTIME_TRAIN_SNAPSHOT_PREFER_SQLITE", "1").strip() == "1")
    parser.add_argument("--reuse-if-fresh-minutes", type=int, default=int(os.getenv("RUNTIME_TRAIN_SNAPSHOT_REUSE_IF_FRESH_MINUTES", "360")))
    parser.add_argument("--rows-path", default=str(DEFAULT_ROWS_PATH))
    parser.add_argument("--health-path", default=str(DEFAULT_HEALTH_PATH))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--seed-health-path", default=os.getenv("RUNTIME_TRAIN_SNAPSHOT_SEED_HEALTH_PATH", ""))
    parser.add_argument("--max-observation-rows", type=int, default=int(os.getenv("RUNTIME_TRAIN_SNAPSHOT_MAX_OBSERVATION_ROWS", "80000")))
    parser.add_argument("--max-sequences", type=int, default=int(os.getenv("RUNTIME_TRAIN_SNAPSHOT_MAX_SEQUENCES", "1200")))
    parser.add_argument("--max-rows-per-sequence", type=int, default=int(os.getenv("RUNTIME_TRAIN_SNAPSHOT_MAX_ROWS_PER_SEQUENCE", "600")))
    parser.add_argument(
        "--incremental-max-runtime-seconds",
        type=float,
        default=_env_float("RUNTIME_TRAIN_INCREMENTAL_MAX_RUNTIME_SECONDS", 180.0),
        help="Maximum seconds to scan incremental JSONL candidates before committing a partial refresh.",
    )
    parser.add_argument(
        "--incremental-max-candidate-rows",
        type=int,
        default=_env_int("RUNTIME_TRAIN_INCREMENTAL_MAX_CANDIDATE_ROWS", 25000),
        help="Maximum valid JSONL candidate rows to parse during incremental refresh; 0 disables the row cap.",
    )
    parser.add_argument("--light-refresh-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    rows_path = Path(args.rows_path).expanduser()
    health_path = Path(args.health_path).expanduser()
    lock_path = Path(args.lock_path).expanduser()
    default_seed_path = Path(DEFAULT_HEALTH_PATH).expanduser()
    seed_health_path = Path(args.seed_health_path).expanduser() if str(args.seed_health_path).strip() else None
    if seed_health_path is None and health_path != default_seed_path:
        seed_health_path = default_seed_path
    mode_allowlist = _parse_csv(args.mode_allowlist)
    symbol_allowlist = _parse_csv(args.symbol_allowlist)
    lock_handle, already_running = _acquire_single_flight_lock(
        lock_path,
        project_root=project_root,
        health_path=health_path,
        rows_path=rows_path,
    )
    if already_running:
        if args.json:
            print(json.dumps(already_running, ensure_ascii=True))
        else:
            print(
                "runtime_training_snapshot already_running=1 "
                f"lock_path={already_running.get('lock_path', '')}"
            )
        return 0
    _ = lock_handle

    current_summary = _load_json(health_path)
    reusable = _reusable_snapshot_payload(
        current_summary,
        project_root=project_root,
        lookback_days=max(int(args.lookback_days), 1),
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
        prefer_sqlite=bool(args.prefer_sqlite),
        max_age_minutes=max(int(args.reuse_if_fresh_minutes), 0),
    )
    if reusable:
        if args.json:
            print(json.dumps(reusable, ensure_ascii=True))
        else:
            print(
                f"runtime_training_snapshot reused=1 sequences={int(reusable.get('sequence_count', 0) or 0)} "
                f"rows={int(reusable.get('row_count', 0) or 0)} rows_path={reusable.get('rows_path', '')}"
            )
        return 0

    if args.light_refresh_existing:
        light_refresh = _light_refresh_existing_snapshot_payload(
            current_summary,
            project_root=project_root,
            health_path=health_path,
            lookback_days=max(int(args.lookback_days), 1),
            mode_allowlist=mode_allowlist,
            symbol_allowlist=symbol_allowlist,
            prefer_sqlite=bool(args.prefer_sqlite),
        )
        if light_refresh:
            health_path.parent.mkdir(parents=True, exist_ok=True)
            health_path.write_text(json.dumps(light_refresh, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if args.json:
                print(json.dumps(light_refresh, ensure_ascii=True))
            else:
                print(
                    f"runtime_training_snapshot light_refresh=1 sequences={int(light_refresh.get('sequence_count', 0) or 0)} "
                    f"rows={int(light_refresh.get('row_count', 0) or 0)} rows_path={light_refresh.get('rows_path', '')}"
                )
            return 0

    incremental_meta: dict[str, Any] = {}
    incremental = _incremental_snapshot_sequences(
        current_summary,
        project_root=project_root,
        health_path=health_path,
        lookback_days=max(int(args.lookback_days), 1),
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
        prefer_sqlite=bool(args.prefer_sqlite),
        max_runtime_seconds=max(float(args.incremental_max_runtime_seconds), 0.0),
        max_candidate_rows=max(int(args.incremental_max_candidate_rows), 0),
    )
    if incremental is not None:
        sequences, incremental_meta = incremental
    else:
        seed_summary = _load_json(seed_health_path) if seed_health_path and seed_health_path.exists() else {}
        seeded = (
            _seeded_snapshot_sequences(
                seed_summary,
                seed_health_path=seed_health_path,
                project_root=project_root,
                lookback_days=max(int(args.lookback_days), 1),
                mode_allowlist=mode_allowlist,
                symbol_allowlist=symbol_allowlist,
            )
            if seed_health_path and seed_health_path.exists()
            else None
        )
        if seeded is not None:
            sequences, incremental_meta = seeded
        else:
            sequences, incremental_meta = _full_refresh_sequences(
                project_root,
                lookback_days=max(int(args.lookback_days), 1),
                mode_allowlist=mode_allowlist,
                symbol_allowlist=symbol_allowlist,
                prefer_sqlite=bool(args.prefer_sqlite),
                max_observation_rows=max(int(args.max_observation_rows), 0),
            )

    sequences = _limit_snapshot_sequences(
        sequences,
        max_sequences=max(int(args.max_sequences), 0),
        max_rows_per_sequence=max(int(args.max_rows_per_sequence), 0),
    )

    rows_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    sequence_count = 0
    coverage = _coverage_summary(sequences)
    with rows_path.open("w", encoding="utf-8") as handle:
        for (mode, symbol), rows in sorted(sequences.items()):
            sequence_count += 1
            for row in rows:
                handle.write(json.dumps({"mode": mode, "symbol": symbol, **row}, ensure_ascii=True) + "\n")
                row_count += 1
    del sequences
    gc.collect()

    payload: dict[str, Any] = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "lookback_days": int(args.lookback_days),
        "mode_allowlist": mode_allowlist,
        "symbol_allowlist": symbol_allowlist,
        "prefer_sqlite": bool(args.prefer_sqlite),
        "max_observation_rows": max(int(args.max_observation_rows), 0),
        "max_sequences": max(int(args.max_sequences), 0),
        "max_rows_per_sequence": max(int(args.max_rows_per_sequence), 0),
        "rows_path": str(rows_path),
        "health_path": str(health_path),
        "lock_path": str(lock_path),
        "jsonl_discovery_manifest": str(project_root / "governance" / "health" / "jsonl_discovery_manifest_latest.json"),
        "rows_sha256": _sha256_file(rows_path),
        "sequence_count": int(sequence_count),
        "row_count": int(row_count),
        "coverage": coverage,
        "single_flight_contract": {
            "active": True,
            "prevents_duplicate_snapshot_builders": True,
            "lock_path": str(lock_path),
        },
    }
    payload.update(_snapshot_content_freshness(payload))
    payload.update(incremental_meta)
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"runtime_training_snapshot sequences={sequence_count} rows={row_count} rows_path={rows_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
