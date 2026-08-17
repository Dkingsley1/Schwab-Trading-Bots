import argparse
import hashlib
import json
import os
import re
import random
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sqlite_runtime import connect_sqlite
from core.runtime_maintenance import maintenance_hold_snapshot, maintenance_hold_token_authorized
from scripts import ops_data_plane

DEFAULT_INCLUDE_GLOBS = [
    "paper_trades_*.jsonl",
    "live_orders_*.jsonl",
    "exports/trade_logs/**/*.jsonl",
    "decision_explanations/**/*.jsonl",
    "decisions/**/*.jsonl",
    "governance/**/*.jsonl",
    "exports/paper_broker_bridge/**/*.jsonl",
    "data/**/*.jsonl",
]
DEFAULT_EXCLUDE_PARTS = ["/.git/", "/.venv", "/models/archive/"]
DEFAULT_INCLUDE_JSON_GLOBS = [
    "master_bot_registry.json",
    "config/**/*.json",
    "governance/health/**/*.json",
    "governance/feature_store/**/*.json",
    "governance/walk_forward/**/*.json",
    "governance/distillation/**/*.json",
    "governance/canary/**/*.json",
    "data/external_context/**/*.json",
    "data/trade_history/**/*.json",
    "exports/external_context/**/*.json",
    "exports/external_feeds/**/*.json",
    "exports/state_snapshot_drills/latest.json",
]
DEFAULT_JSON_EXCLUDE_PARTS = ["/.git/", "/.venv", "/models/archive/", "/exports/reports/"]
DISCOVERY_MANIFEST_SCHEMA_VERSION = 1


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _host_load_1m() -> float:
    try:
        return max(float(os.getloadavg()[0]), 0.0)
    except Exception:
        return 0.0


def _ingest_cooldown_sleep(
    *,
    base_sleep_seconds: float,
    host_load_soft_cap: float,
    host_load_sleep_seconds: float,
) -> float:
    sleep_seconds = max(float(base_sleep_seconds), 0.0)
    load_cap = max(float(host_load_soft_cap), 0.0)
    if load_cap > 0.0 and _host_load_1m() >= load_cap:
        sleep_seconds = max(sleep_seconds, max(float(host_load_sleep_seconds), 0.0))
    if sleep_seconds <= 0.0:
        return 0.0
    time.sleep(sleep_seconds)
    return sleep_seconds


def _log_schema_version() -> int:
    try:
        return max(int(os.getenv("LOG_SCHEMA_VERSION", "2")), 1)
    except Exception:
        return 2


def _classify_stream(source_rel: str) -> str:
    rel = str(source_rel or "")
    if rel.startswith("decision_explanations/"):
        return "decision_explanations"
    if rel.startswith("decisions/"):
        return "decisions"
    if rel.startswith("governance/channels/decision/"):
        return "decisions"
    if rel.startswith("governance/events/channel_schema_violations_"):
        return "schema_violations"
    if rel.startswith("governance/events/"):
        return "governance_events"
    if rel.startswith("governance/watchdog/"):
        return "governance_watchdog"
    if rel.startswith("governance/"):
        return "governance"
    if rel.startswith("exports/trade_logs/"):
        return "trade_logs"
    if rel.startswith("exports/paper_broker_bridge/"):
        return "paper_broker_bridge"
    if rel.startswith("paper_trades_") or rel.startswith("live_orders_"):
        return "top_level_trade_links"
    if rel.startswith("data/"):
        return "data"
    return "other"


def _source_priority(source_rel: str) -> int:
    stream = _classify_stream(source_rel)
    weights = {
        "decisions": 0,
        "paper_broker_bridge": 1,
        "top_level_trade_links": 2,
        "trade_logs": 3,
        "governance_events": 4,
        "governance_watchdog": 5,
        "decision_explanations": 6,
        "governance": 7,
        "schema_violations": 8,
        "data": 9,
        "other": 10,
    }
    return int(weights.get(stream, 10))


def _path_hot_priority(source_rel: str) -> int:
    rel = str(source_rel or "")
    if rel.startswith("decisions/"):
        return 0
    if rel.startswith("governance/channels/decision/"):
        return 1
    if rel.startswith("exports/paper_broker_bridge/") or rel.startswith("paper_trades_") or rel.startswith("live_orders_"):
        return 2
    if rel.startswith("decision_explanations/"):
        return 3
    if rel.startswith("governance/events/channel_schema_violations_"):
        return 13
    hot_prefixes = (
        ("governance/events/gate_logs_", 4),
        ("governance/channels/gate/", 5),
        ("governance/channels/risk/", 6),
        ("governance/shadow_", 7),
        ("governance/channels/api/", 8),
        ("governance/channels/ingress/", 9),
        ("governance/channels/runtime/", 10),
    )
    for prefix, priority in hot_prefixes:
        if rel.startswith(prefix):
            if prefix == "governance/shadow_" and "/shadow_pnl_attribution_" in rel:
                return 12
            return priority
    return 11


def _storage_temperature_label(source_rel: str) -> str:
    rel = str(source_rel or "")
    if rel.startswith("decisions/"):
        return "hot"
    if rel.startswith("exports/paper_broker_bridge/") or rel.startswith("paper_trades_") or rel.startswith("live_orders_"):
        return "hot"
    if rel.startswith("governance/events/gate_logs_") or rel.startswith("governance/channels/decision/"):
        return "hot"
    if _is_cold_lane_path(rel):
        return "cold"
    if rel.startswith("decision_explanations/"):
        return "warm"
    if _is_deferred_analytics_path(rel):
        return "warm"
    if rel.startswith("data/"):
        return "cool"
    return "warm"


def _storage_tier_label(source_rel: str) -> str:
    temp = _storage_temperature_label(source_rel)
    if temp == "hot":
        return "primary_hot"
    if temp == "warm":
        return "primary_warm"
    if temp == "cool":
        return "compatibility_cool"
    return "archive_cold"


def _ingestion_lane_label(source_rel: str) -> str:
    rel = str(source_rel or "")
    if _is_cold_lane_path(rel):
        return "cold_lane"
    if rel.startswith("decisions/") or rel.startswith("exports/paper_broker_bridge/") or rel.startswith("paper_trades_") or rel.startswith("live_orders_"):
        return "hot_lane"
    if _is_deferred_analytics_path(rel):
        return "deferred_lane"
    return "nearline_lane"


def _stale_age_bucket(age_seconds: float) -> str:
    age = max(float(age_seconds), 0.0)
    if age < 15 * 60:
        return "fresh_lt_15m"
    if age < 2 * 60 * 60:
        return "recent_lt_2h"
    if age < 24 * 60 * 60:
        return "aging_lt_24h"
    if age < 7 * 24 * 60 * 60:
        return "stale_lt_7d"
    return "cold_gte_7d"


def _filename_date_rank(source_rel: str, *, anchor_day: Optional[date] = None) -> int:
    rel = str(source_rel or "")
    match = re.search(r"(20\d{6})", rel)
    if not match:
        return 4
    raw = match.group(1)
    try:
        file_day = datetime.strptime(raw, "%Y%m%d").date()
    except Exception:
        return 4
    base_day = anchor_day or datetime.now(timezone.utc).date()
    delta_days = max((base_day - file_day).days, 0)
    if delta_days <= 0:
        return 0
    if delta_days == 1:
        return 1
    if delta_days <= 3:
        return 2
    if delta_days <= 7:
        return 3
    return 4


def _is_deferred_analytics_path(source_rel: str) -> bool:
    rel = str(source_rel or "")
    return (
        rel.startswith("governance/events/api_calls_")
        or rel.startswith("governance/events/data_ingress_")
        or rel.startswith("governance/channels/api/")
        or rel.startswith("governance/channels/ingress/")
        or rel.startswith("governance/channels/runtime/")
        or "/shadow_pnl_attribution_" in rel
    )


def _is_cold_lane_path(source_rel: str) -> bool:
    rel = str(source_rel or "")
    return "/shadow_pnl_attribution_" in rel


def _parse_csv_values(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if str(part).strip()]


def _discovery_manifest_path(project_root: Path, kind: str) -> Path:
    suffix = "jsonl" if str(kind).strip().lower() != "json" else "json"
    return project_root / "governance" / "health" / f"{suffix}_discovery_manifest_latest.json"


def _discovery_manifest_max_age_seconds(kind: str) -> float:
    if str(kind).strip().lower() == "json":
        env_name = "JSON_DISCOVERY_MANIFEST_MAX_AGE_SECONDS"
    else:
        env_name = "JSONL_DISCOVERY_MANIFEST_MAX_AGE_SECONDS"
    try:
        return max(float(os.getenv(env_name, "300")), 0.0)
    except Exception:
        return 300.0


def _scan_matching_files(project_root: Path, include: List[str], excludes: List[str]) -> List[Path]:
    found: List[Path] = []
    seen_path = set()
    seen_resolved = set()

    for pat in include:
        for p in project_root.glob(pat):
            if not p.is_file():
                continue
            p_str = str(p)
            if any(part and part in p_str for part in excludes):
                continue
            try:
                resolved = str(p.resolve(strict=False))
            except Exception:
                resolved = p_str
            if p_str in seen_path or resolved in seen_resolved:
                continue
            seen_path.add(p_str)
            seen_resolved.add(resolved)
            found.append(p)
    return found


_HOT_CHANNEL_DUPLICATE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("api_calls_", "api", "api"),
    ("data_ingress_", "ingress", "ingress"),
    ("loop_state_", "loop_state", "loop_state"),
    ("gate_logs_", "gate", "gate"),
)


def _prefer_channel_primary_logs() -> bool:
    return str(os.getenv("PREFER_CHANNEL_PRIMARY_LOGS", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}


def _is_redundant_hot_channel_duplicate(project_root: Path, path: Path) -> bool:
    if not _prefer_channel_primary_logs():
        return False
    try:
        rel = str(path.relative_to(project_root)).replace("\\", "/")
    except Exception:
        rel = str(path).replace("\\", "/")
    if not rel.startswith("governance/events/"):
        return False

    name = path.name
    runtime_match = re.fullmatch(r"runtime_events_(20\d{6})\.jsonl", name)
    if runtime_match:
        day = runtime_match.group(1)
        return any((project_root / "governance" / "channels" / "runtime").glob(f"*/runtime_{day}.jsonl"))

    for legacy_prefix, channel_name, channel_file_prefix in _HOT_CHANNEL_DUPLICATE_PATTERNS:
        match = re.fullmatch(rf"{re.escape(legacy_prefix)}(.+)_(20\d{{6}})\.jsonl", name)
        if not match:
            continue
        _context_key, day = match.groups()
        return any((project_root / "governance" / "channels" / channel_name).glob(f"*/{channel_file_prefix}_{day}.jsonl"))
    return False


def _filter_redundant_hot_channel_duplicates(project_root: Path, files: List[Path]) -> List[Path]:
    return [path for path in files if not _is_redundant_hot_channel_duplicate(project_root, path)]


def _load_discovery_manifest(
    *,
    project_root: Path,
    include_globs: List[str],
    exclude_parts: List[str],
    kind: str,
) -> Optional[List[Path]]:
    path = _discovery_manifest_path(project_root, kind)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("schema_version", 0) or 0) != DISCOVERY_MANIFEST_SCHEMA_VERSION:
        return None
    if list(payload.get("include_globs") or []) != list(include_globs):
        return None
    if list(payload.get("exclude_parts") or []) != list(exclude_parts):
        return None
    manifest_ts = str(payload.get("timestamp_utc") or "").strip()
    if not manifest_ts:
        return None
    try:
        manifest_age = max(
            datetime.now(timezone.utc).timestamp()
            - datetime.fromisoformat(manifest_ts.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp(),
            0.0,
        )
    except Exception:
        return None
    if manifest_age > _discovery_manifest_max_age_seconds(kind):
        return None

    rows = payload.get("files") if isinstance(payload.get("files"), list) else []
    out: List[Path] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        rel = str(row.get("source_rel") or "").strip()
        if not rel:
            return None
        path_obj = project_root / rel
        if not path_obj.is_file():
            return None
        try:
            stat = path_obj.stat()
        except Exception:
            return None
        if int(row.get("size_bytes", -1) or -1) != int(stat.st_size):
            return None
        try:
            manifest_mtime = float(row.get("mtime_epoch", 0.0) or 0.0)
        except Exception:
            return None
        if abs(float(stat.st_mtime) - manifest_mtime) > 1e-6:
            return None
        out.append(path_obj)
    return out


def _write_discovery_manifest(
    *,
    project_root: Path,
    include_globs: List[str],
    exclude_parts: List[str],
    kind: str,
    files: List[Path],
) -> None:
    rows: List[Dict[str, Any]] = []
    for path in files:
        try:
            rel = str(path.relative_to(project_root))
        except Exception:
            rel = str(path)
        try:
            stat = path.stat()
            size_bytes = int(stat.st_size)
            mtime_epoch = float(stat.st_mtime)
        except Exception:
            size_bytes = 0
            mtime_epoch = 0.0
        rows.append(
            {
                "source_rel": rel,
                "size_bytes": int(size_bytes),
                "mtime_epoch": float(mtime_epoch),
                "source_priority": int(_source_priority(rel)),
                "temperature": _storage_temperature_label(rel),
                "storage_tier": _storage_tier_label(rel),
                "ingestion_lane": _ingestion_lane_label(rel),
            }
        )

    payload = {
        "timestamp_utc": _now_utc(),
        "schema_version": DISCOVERY_MANIFEST_SCHEMA_VERSION,
        "project_root": str(project_root),
        "kind": str(kind),
        "include_globs": list(include_globs),
        "exclude_parts": list(exclude_parts),
        "file_count": len(rows),
        "total_bytes": int(sum(int(row.get("size_bytes", 0) or 0) for row in rows)),
        "files": rows,
    }
    path = _discovery_manifest_path(project_root, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _json_file_priority(source_rel: str) -> int:
    stream = _json_file_stream(source_rel)
    weights = {
        "registry": 0,
        "config": 1,
        "external_context": 2,
        "external_feeds": 3,
        "feature_store": 4,
        "event_store": 5,
        "governance_health": 6,
        "governance_walk_forward": 7,
        "governance_distillation": 8,
        "governance_canary": 9,
        "trade_history_json": 10,
        "state_snapshot_drill": 11,
        "json_file": 12,
    }
    return int(weights.get(stream, 12))


def _matches_rel_filters(
    *,
    source_rel: str,
    stream: str,
    include_streams: List[str],
    exclude_streams: List[str],
    path_contains: List[str],
    path_not_contains: List[str],
) -> bool:
    rel = str(source_rel or "")
    include_streams_set = set(include_streams)
    exclude_streams_set = set(exclude_streams)
    if include_streams_set and stream not in include_streams_set:
        return False
    if exclude_streams_set and stream in exclude_streams_set:
        return False
    if path_contains and not any(token in rel for token in path_contains):
        return False
    if path_not_contains and any(token in rel for token in path_not_contains):
        return False
    return True


def discover_jsonl_files(
    project_root: Path,
    include_globs: Optional[List[str]] = None,
    exclude_parts: Optional[List[str]] = None,
) -> List[Path]:
    include = include_globs or list(DEFAULT_INCLUDE_GLOBS)
    excludes = exclude_parts or list(DEFAULT_EXCLUDE_PARTS)

    manifest_rows = _load_discovery_manifest(
        project_root=project_root,
        include_globs=include,
        exclude_parts=excludes,
        kind="jsonl",
    )
    if manifest_rows is not None:
        manifest_rows = _filter_redundant_hot_channel_duplicates(project_root, manifest_rows)
    current_rows = _scan_matching_files(project_root, include, excludes)
    current_rows = _filter_redundant_hot_channel_duplicates(project_root, current_rows)
    if manifest_rows is not None:
        manifest_rels = {
            str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path)
            for path in manifest_rows
        }
        current_rels = {
            str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path)
            for path in current_rows
        }
        if manifest_rels == current_rels:
            return manifest_rows

    found = current_rows
    if manifest_rows is not None and not found:
        return manifest_rows

    def _sort_key(path: Path) -> Tuple[int, float, str]:
        try:
            rel = str(path.relative_to(project_root))
        except Exception:
            rel = str(path)
        try:
            mtime = float(path.stat().st_mtime)
        except Exception:
            mtime = 0.0
        return (_source_priority(rel), -mtime, rel)

    found.sort(key=_sort_key)
    _write_discovery_manifest(
        project_root=project_root,
        include_globs=include,
        exclude_parts=excludes,
        kind="jsonl",
        files=found,
    )
    return found


def _discover_jsonl_files(project_root: Path, include_globs: List[str], exclude_parts: List[str]) -> List[Path]:
    # Backward-compatible helper kept for tests and internal callers.
    return discover_jsonl_files(project_root, include_globs=include_globs, exclude_parts=exclude_parts)


def discover_json_files(
    project_root: Path,
    include_globs: Optional[List[str]] = None,
    exclude_parts: Optional[List[str]] = None,
) -> List[Path]:
    include = include_globs or list(DEFAULT_INCLUDE_JSON_GLOBS)
    excludes = exclude_parts or list(DEFAULT_JSON_EXCLUDE_PARTS)

    found: List[Path] = []
    seen_path = set()
    seen_resolved = set()

    for pat in include:
        for p in project_root.glob(pat):
            if not p.is_file():
                continue
            p_str = str(p)
            if any(part and part in p_str for part in excludes):
                continue
            try:
                resolved = str(p.resolve(strict=False))
            except Exception:
                resolved = p_str
            if p_str in seen_path or resolved in seen_resolved:
                continue
            seen_path.add(p_str)
            seen_resolved.add(resolved)
            found.append(p)

    found.sort(key=lambda path: str(path.relative_to(project_root)))
    return found


def _prioritize_jsonl_files_by_pending_bytes(
    files: List[Path],
    *,
    project_root: Path,
    sqlite_state: Dict[str, Dict[str, Any]],
) -> List[Path]:
    anchor_days: List[datetime.date] = []
    for path in files:
        try:
            rel = str(path.relative_to(project_root))
        except Exception:
            rel = str(path)
        match = re.search(r"(20\d{6})", rel)
        if not match:
            continue
        try:
            anchor_days.append(datetime.strptime(match.group(1), "%Y%m%d").date())
        except Exception:
            continue
    filename_anchor_day = max(anchor_days) if anchor_days else datetime.now(timezone.utc).date()

    def _sort_key(path: Path) -> Tuple[int, int, int, int, int, int, int, int, float, str]:
        try:
            rel = str(path.relative_to(project_root))
        except Exception:
            rel = str(path)
        progress = sqlite_state.get(rel, {}) if isinstance(sqlite_state, dict) else {}
        try:
            stat = path.stat()
            size_bytes = int(stat.st_size)
            mtime = float(stat.st_mtime)
        except Exception:
            size_bytes = 0
            mtime = 0.0
        last_offset = int(float(progress.get("last_offset_bytes", 0) or 0))
        pending_bytes = max(size_bytes - max(last_offset, 0), 0)
        has_pending = 0 if pending_bytes > 0 else 1
        lane = _ingestion_lane_label(rel)
        lane_rank = {
            "hot_lane": 0,
            "nearline_lane": 1,
            "deferred_lane": 2,
            "cold_lane": 3,
        }.get(lane, 1)
        temperature = _storage_temperature_label(rel)
        temperature_rank = {
            "hot": 0,
            "warm": 1,
            "cool": 2,
            "cold": 3,
        }.get(temperature, 1)
        filename_date_rank = _filename_date_rank(rel, anchor_day=filename_anchor_day)
        age_bucket = _stale_age_bucket(max(time.time() - max(mtime, 0.0), 0.0))
        # Within the same hot lane and date bucket, give older pending files a
        # fairness turn before fresh high-volume files keep winning the cap.
        stale_rank = {
            "cold_gte_7d": 0,
            "stale_lt_7d": 1,
            "aging_lt_24h": 2,
            "recent_lt_2h": 3,
            "fresh_lt_15m": 4,
        }.get(age_bucket, 2)
        return (
            has_pending,
            lane_rank,
            temperature_rank,
            filename_date_rank,
            stale_rank,
            _path_hot_priority(rel),
            -pending_bytes,
            _source_priority(rel),
            -mtime,
            rel,
        )

    return sorted(files, key=_sort_key)


def _limit_prioritized_jsonl_files(
    files: List[Path],
    *,
    project_root: Path,
    max_files: int,
    max_deferred_files: int,
) -> List[Path]:
    if max_files <= 0:
        return list(files)
    core_files: List[Path] = []
    deferred_files: List[Path] = []
    cold_lane_files: List[Path] = []
    for path in files:
        try:
            rel = str(path.relative_to(project_root))
        except Exception:
            rel = str(path)
        if _is_cold_lane_path(rel):
            cold_lane_files.append(path)
        elif _is_deferred_analytics_path(rel):
            deferred_files.append(path)
        else:
            core_files.append(path)
    if not core_files:
        if cold_lane_files:
            return cold_lane_files[:max_files]
        return deferred_files[:max_files]

    kept = core_files[:max_files]
    remaining = max(max_files - len(kept), 0)
    if remaining <= 0 or max_deferred_files <= 0:
        return kept
    kept.extend(deferred_files[: min(remaining, max_deferred_files)])
    remaining = max(max_files - len(kept), 0)
    cold_budget = max(int(os.getenv("JSONL_SQL_MAX_COLD_LANE_FILES", "0") or 0), 0)
    if remaining > 0 and cold_budget > 0:
        kept.extend(cold_lane_files[: min(remaining, cold_budget)])
    return kept


def _load_state(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return {"sqlite": {}, "mysql": {}, "sqlite_json_files": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {"sqlite": {}, "mysql": {}, "sqlite_json_files": {}}
        obj.setdefault("sqlite", {})
        obj.setdefault("mysql", {})
        obj.setdefault("sqlite_json_files", {})
        return obj
    except Exception:
        return {"sqlite": {}, "mysql": {}, "sqlite_json_files": {}}


def _save_state(path: Path, state: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    except Exception:
        return


def _parse_ts_utc(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_event_ts_utc(obj: Dict[str, Any]) -> Optional[datetime]:
    md = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    keys = [
        "timestamp_utc",
        "ts_utc",
        "event_timestamp_utc",
        "created_at",
        "timestamp",
    ]
    for key in keys:
        dt = _parse_ts_utc(obj.get(key))
        if dt is not None:
            return dt
    for key in keys:
        dt = _parse_ts_utc(md.get(key))
        if dt is not None:
            return dt
    return None


def _iter_new_lines(path: Path, start_line: int, start_offset_bytes: int = 0) -> Iterable[Tuple[int, str, int]]:
    line_no = max(int(start_line), 0)
    offset = max(int(start_offset_bytes), 0)

    with open(path, "rb") as f:
        if offset > 0:
            try:
                f.seek(offset)
            except Exception:
                f.seek(0)
                line_no = 0

        while True:
            raw = f.readline()
            if not raw:
                break
            line_no += 1
            out_offset = int(f.tell())
            line = raw.rstrip(b"\r\n")
            if not line.strip():
                continue
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                text = line.decode("utf-8", errors="replace")
            yield line_no, text, out_offset


def _extract_correlation_fields(obj: Dict[str, Any]) -> Tuple[str, str, str, str, int]:
    md = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

    def _pick(key: str) -> str:
        val = obj.get(key)
        if val is None or val == "":
            val = md.get(key)
        return str(val) if (val is not None and val != "") else ""

    run_id = _pick("run_id")
    iter_id = _pick("iter_id")
    decision_id = _pick("decision_id")
    parent_decision_id = _pick("parent_decision_id")

    raw_schema = obj.get("log_schema_version")
    if raw_schema is None:
        raw_schema = md.get("log_schema_version")
    try:
        schema_version = max(int(raw_schema), 0)
    except Exception:
        schema_version = 0
    if schema_version <= 0:
        schema_version = _log_schema_version()

    return run_id, iter_id, decision_id, parent_decision_id, schema_version


def _route_text(value: Any) -> str:
    return str(value or "").strip()


def _route_hint_text(obj: Dict[str, Any], route: Dict[str, Any], metadata: Dict[str, Any], *, source_rel: str) -> Tuple[str, str]:
    path_hints = [
        source_rel,
        obj.get("source_path"),
        obj.get("target_path"),
        obj.get("file_path"),
        obj.get("path"),
        obj.get("decision_path"),
        metadata.get("source_path"),
        metadata.get("target_path"),
        metadata.get("file_path"),
        route.get("source_path"),
        route.get("route_key"),
    ]
    context_hints = [
        obj.get("event"),
        obj.get("strategy"),
        obj.get("profile"),
        obj.get("shadow_profile"),
        obj.get("domain"),
        obj.get("shadow_domain"),
        obj.get("source_stream"),
        obj.get("source_partition_key"),
        route.get("channel"),
        route.get("profile"),
        route.get("domain"),
        metadata.get("event"),
        metadata.get("strategy"),
        metadata.get("source_stream"),
    ]
    path_text = " ".join(str(item or "") for item in path_hints if str(item or "").strip()).lower()
    context_text = " ".join(str(item or "") for item in context_hints if str(item or "").strip()).lower()
    return path_text, " ".join(part for part in (path_text, context_text) if part).lower()


def _extract_route_fields(obj: Dict[str, Any], *, source_rel: str = "") -> Tuple[str, str, str, str, str, str]:
    route = obj.get("data_route") if isinstance(obj.get("data_route"), dict) else {}
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

    def _pick(*keys: str) -> str:
        for key in keys:
            for bucket in (obj, route, metadata):
                value = bucket.get(key) if isinstance(bucket, dict) else None
                text = _route_text(value)
                if text:
                    return text
        return ""

    source_broker = _pick("source_broker", "broker")
    source_provider = _pick("source_provider", "provider", "source")
    source_venue = _pick("source_venue", "venue")
    asset_class = _pick("asset_class", "market_kind", "instrument_class", "domain")
    routing_lane = _pick("routing_lane", "lane")
    source_quality_label = _pick("source_quality_label", "quality_label")

    symbol = str(obj.get("symbol") or obj.get("underlying_symbol") or "").strip().upper()
    path_haystack, route_haystack = _route_hint_text(obj, route, metadata, source_rel=source_rel)
    haystack = " ".join([route_haystack, source_broker, source_provider, source_venue, asset_class, routing_lane, symbol]).lower()
    if not source_broker:
        if "coinbase" in haystack:
            source_broker = "coinbase"
        elif "schwab" in haystack:
            source_broker = "schwab"
        elif "crypto" not in haystack and ("equities" in path_haystack or "equity" in path_haystack):
            source_broker = "schwab"
    if not source_provider:
        if "coinbase" in haystack:
            source_provider = "coinbase"
        elif "schwab_crypto" in haystack:
            source_provider = "schwab_crypto"
        elif "schwab" in haystack:
            source_provider = "schwab"
        else:
            source_provider = source_broker
    if not source_venue:
        if "schwab_crypto" in haystack:
            source_venue = "schwab_crypto_bridge"
        elif "coinbase" in haystack:
            source_venue = "coinbase"
        elif "schwab" in haystack:
            source_venue = "schwab"
        else:
            source_venue = source_provider
    if not asset_class:
        if symbol.startswith("/"):
            asset_class = "futures"
        elif (
            "shadow_crypto" in path_haystack
            or "crypto_coinbase" in path_haystack
            or source_broker == "coinbase"
            or symbol.endswith(("-USD", "-USDT", "-USDC"))
        ):
            asset_class = "crypto"
        elif "equities" in path_haystack or "equity" in path_haystack:
            asset_class = "equities"
        elif "schwab_futures" in path_haystack or "_futures" in path_haystack:
            asset_class = "futures"
        elif "option" in haystack:
            asset_class = "options"
        elif "fx" in haystack or "forex" in haystack:
            asset_class = "fx"
    if not routing_lane:
        if source_venue == "schwab_crypto_bridge":
            routing_lane = "schwab_crypto_bridge"
        elif source_broker and asset_class:
            routing_lane = f"{source_broker}_{asset_class}"
    if not source_quality_label:
        if source_venue == "schwab_crypto_bridge":
            source_quality_label = "broker_bridge"
        elif source_broker == "schwab":
            source_quality_label = "broker_native"
        elif source_broker == "coinbase":
            source_quality_label = "exchange_native"

    return (
        source_broker,
        source_provider,
        source_venue,
        asset_class,
        routing_lane,
        source_quality_label,
    )


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _event_day_utc(event_ts: Optional[datetime], *, fallback_ts: datetime) -> str:
    dt = event_ts or fallback_ts
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _derive_start_cursor(progress: Dict[str, Any], stat: os.stat_result) -> Tuple[int, int, str]:
    start_line = int(float(progress.get("last_line", 0) or 0))
    start_offset = int(float(progress.get("last_offset_bytes", 0) or 0))
    prev_mtime = float(progress.get("mtime", 0.0) or 0.0)

    prev_inode = int(float(progress.get("file_inode", 0) or 0))
    prev_size = int(float(progress.get("file_size_bytes", 0) or 0))

    if prev_inode > 0 and int(stat.st_ino) != prev_inode:
        return 0, 0, "inode_changed"
    if prev_size > 0 and int(stat.st_size) < prev_size:
        return 0, 0, "size_shrank"
    if float(stat.st_mtime) < prev_mtime:
        return 0, 0, "mtime_rewound"
    if start_offset > int(stat.st_size):
        return 0, 0, "offset_past_eof"

    return max(start_line, 0), max(start_offset, 0), ""


def _record_top_pending(
    rows: List[Dict[str, Any]],
    *,
    source_rel: str,
    pending_lines: int,
    oldest_age_seconds: float,
    total_lines: int,
    last_line: int,
    top_n: int,
) -> None:
    if pending_lines <= 0:
        return
    rows.append(
        {
            "source_rel": str(source_rel),
            "stream": _classify_stream(source_rel),
            "storage_temperature": _storage_temperature_label(source_rel),
            "storage_tier": _storage_tier_label(source_rel),
            "ingestion_lane": _ingestion_lane_label(source_rel),
            "stale_age_bucket": _stale_age_bucket(oldest_age_seconds),
            "pending_lines": int(pending_lines),
            "oldest_pending_age_seconds": round(float(oldest_age_seconds), 3),
            "total_lines": int(total_lines),
            "last_line": int(last_line),
        }
    )
    rows.sort(
        key=lambda r: (
            int(r.get("pending_lines", 0)),
            float(r.get("oldest_pending_age_seconds", 0.0) or 0.0),
        ),
        reverse=True,
    )
    if len(rows) > max(int(top_n), 1):
        del rows[max(int(top_n), 1) :]


class LatencyAccumulator:
    def __init__(self, reservoir_size: int = 2048) -> None:
        self.reservoir_size = max(int(reservoir_size), 128)
        self.count = 0
        self.total_seconds = 0.0
        self.max_seconds = 0.0
        self.slo_breaches_300s = 0
        self._samples: List[float] = []

    def add(self, latency_seconds: float) -> None:
        val = max(float(latency_seconds), 0.0)
        self.count += 1
        self.total_seconds += val
        if val > self.max_seconds:
            self.max_seconds = val
        if val > 300.0:
            self.slo_breaches_300s += 1

        if len(self._samples) < self.reservoir_size:
            self._samples.append(val)
            return

        idx = random.randint(0, self.count - 1)
        if idx < self.reservoir_size:
            self._samples[idx] = val

    def snapshot(self) -> Dict[str, Any]:
        if self.count <= 0:
            return {
                "count": 0,
                "p50_seconds": 0.0,
                "p95_seconds": 0.0,
                "max_seconds": 0.0,
                "mean_seconds": 0.0,
                "slo_breach_ratio_gt_300s": 0.0,
            }

        vals = sorted(self._samples)

        def _pct(p: float) -> float:
            if not vals:
                return 0.0
            i = min(max(int(round((len(vals) - 1) * p)), 0), len(vals) - 1)
            return float(vals[i])

        return {
            "count": int(self.count),
            "p50_seconds": round(_pct(0.50), 3),
            "p95_seconds": round(_pct(0.95), 3),
            "max_seconds": round(float(self.max_seconds), 3),
            "mean_seconds": round(float(self.total_seconds) / max(int(self.count), 1), 3),
            "slo_breach_ratio_gt_300s": round(float(self.slo_breaches_300s) / max(int(self.count), 1), 6),
        }


def _latency_payload(acc_all: LatencyAccumulator, by_stream: Dict[str, LatencyAccumulator]) -> Dict[str, Any]:
    stream_rows = {}
    for stream, acc in sorted(by_stream.items()):
        snap = acc.snapshot()
        if int(snap.get("count", 0)) > 0:
            stream_rows[str(stream)] = snap
    return {
        "all": acc_all.snapshot(),
        "by_stream": stream_rows,
    }


def _ensure_latency_bucket(
    store: Dict[str, Dict[str, Any]],
    mode: str,
    stream: str,
) -> Tuple[LatencyAccumulator, LatencyAccumulator]:
    mode_obj = store.setdefault(mode, {"all": LatencyAccumulator(), "by_stream": {}})
    by_stream = mode_obj.setdefault("by_stream", {})
    stream_acc = by_stream.get(stream)
    if stream_acc is None:
        stream_acc = LatencyAccumulator()
        by_stream[stream] = stream_acc
    return mode_obj["all"], stream_acc


def _log_invalid_line(
    *,
    invalid_log_path: Optional[Path],
    mode: str,
    source_rel: str,
    line_no: int,
    raw: str,
    error: Exception,
    run_id: str,
    iter_id: str,
) -> None:
    if invalid_log_path is None:
        return
    _append_jsonl(
        invalid_log_path,
        {
            "timestamp_utc": _now_utc(),
            "event": "ingest_invalid_json",
            "mode": str(mode),
            "source_rel": str(source_rel),
            "stream": _classify_stream(source_rel),
            "line_no": int(line_no),
            "error": str(error),
            "raw_sample": str(raw)[:512],
            "run_id": str(run_id or ""),
            "iter_id": str(iter_id or ""),
            "log_schema_version": _log_schema_version(),
        },
    )


def _ensure_sqlite_schema(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            run_id TEXT,
            iter_id TEXT,
            decision_id TEXT,
            parent_decision_id TEXT,
            log_schema_version INTEGER,
            source_day_utc TEXT,
            source_stream TEXT,
            source_partition_key TEXT,
            source_broker TEXT,
            source_provider TEXT,
            source_venue TEXT,
            asset_class TEXT,
            routing_lane TEXT,
            source_quality_label TEXT,
            UNIQUE(source_file, line_no)
        )
        """
    )

    try:
        cols = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    except Exception:
        cols = set()

    expected_cols = {
        "run_id": "TEXT",
        "iter_id": "TEXT",
        "decision_id": "TEXT",
        "parent_decision_id": "TEXT",
        "log_schema_version": "INTEGER",
        "source_day_utc": "TEXT",
        "source_stream": "TEXT",
        "source_partition_key": "TEXT",
        "source_broker": "TEXT",
        "source_provider": "TEXT",
        "source_venue": "TEXT",
        "asset_class": "TEXT",
        "routing_lane": "TEXT",
        "source_quality_label": "TEXT",
    }
    for col, col_type in expected_cols.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_source_rel ON {table}(source_rel)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_source_rel_line ON {table}(source_rel, line_no)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ingested_at ON {table}(ingested_at)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_run_id ON {table}(run_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_iter_id ON {table}(iter_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_decision_id ON {table}(decision_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_source_day_stream ON {table}(source_day_utc, source_stream)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_source_partition_key ON {table}(source_partition_key)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_source_broker ON {table}(source_broker)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_routing_lane ON {table}(routing_lane)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_asset_class ON {table}(asset_class)")


def _json_file_stream(source_rel: str) -> str:
    rel = str(source_rel or "")
    if rel == "master_bot_registry.json":
        return "registry"
    if rel.startswith("config/"):
        return "config"
    if rel.startswith("data/external_context/") or rel.startswith("exports/external_context/"):
        return "external_context"
    if rel.startswith("exports/external_feeds/"):
        return "external_feeds"
    if rel.startswith("governance/feature_store/"):
        return "feature_store"
    if rel.startswith("governance/health/point_in_time_event_store"):
        return "event_store"
    if rel.startswith("governance/health/"):
        return "governance_health"
    if rel.startswith("governance/walk_forward/"):
        return "governance_walk_forward"
    if rel.startswith("governance/distillation/"):
        return "governance_distillation"
    if rel.startswith("governance/canary/"):
        return "governance_canary"
    if rel.startswith("data/trade_history/"):
        return "trade_history_json"
    if rel.startswith("exports/state_snapshot_drills/"):
        return "state_snapshot_drill"
    return "json_file"


def _ensure_sqlite_json_file_schema(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            stream TEXT NOT NULL,
            modified_at TEXT,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_size_bytes INTEGER NOT NULL DEFAULT 0,
            log_schema_version INTEGER,
            UNIQUE(source_rel, payload_sha1)
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_source_rel ON {table}(source_rel)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_stream ON {table}(stream)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ingested_at ON {table}(ingested_at)")


def _sync_json_file_to_sqlite(
    conn: Optional[sqlite3.Connection],
    table: str,
    project_root: Path,
    file_path: Path,
    dry_run: bool,
    lock_retries: int,
    lock_retry_delay_seconds: float,
) -> Dict[str, Any]:
    source_rel = str(file_path.relative_to(project_root))
    modified_at = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()
    obj = json.loads(file_path.read_text(encoding="utf-8"))
    payload_json = json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    payload_sha1 = hashlib.sha1(payload_json.encode("utf-8")).hexdigest()
    row = (
        str(file_path),
        source_rel,
        _json_file_stream(source_rel),
        modified_at,
        _now_utc(),
        payload_sha1,
        payload_json,
        len(payload_json.encode("utf-8")),
        _log_schema_version(),
    )
    inserted = 0
    if not dry_run:
        if conn is None:
            raise RuntimeError("sqlite connection missing")
        cur = _sqlite_executemany_with_retry(
            conn,
            f"INSERT OR IGNORE INTO {table} "
            "(source_file, source_rel, stream, modified_at, ingested_at, payload_sha1, payload_json, payload_size_bytes, log_schema_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [row],
            lock_retries=lock_retries,
            lock_retry_delay_seconds=lock_retry_delay_seconds,
        )
        inserted = cur.rowcount if cur.rowcount is not None else 0
    else:
        inserted = 1
    return {
        "inserted": int(inserted),
        "payload_sha1": payload_sha1,
        "payload_size_bytes": len(payload_json.encode("utf-8")),
        "stream": _json_file_stream(source_rel),
    }


def _sqlite_executemany_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    rows: List[Tuple[Any, ...]],
    lock_retries: int,
    lock_retry_delay_seconds: float,
) -> sqlite3.Cursor:
    attempt = 0
    while True:
        try:
            return conn.executemany(sql, rows)
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            is_locked = ("database is locked" in msg) or ("database table is locked" in msg)
            if (not is_locked) or attempt >= max(lock_retries, 0):
                raise
            sleep_s = min(max(lock_retry_delay_seconds, 0.01) * (2 ** attempt), 5.0)
            print(
                f"SQLite busy; retrying batch in {sleep_s:.2f}s "
                f"(attempt {attempt + 1}/{max(lock_retries, 0)})"
            )
            time.sleep(sleep_s)
            attempt += 1


def _source_file_identity_for_sqlite_insert(
    conn: Optional[sqlite3.Connection],
    table: str,
    file_path: Path,
    source_rel: str,
    start_line: int,
) -> str:
    source_file = str(file_path)
    rel = str(source_rel or "").replace("\\", "/")
    if conn is None or int(start_line) != 0 or not rel.startswith("governance/channels/"):
        return source_file
    try:
        existing = conn.execute(
            f"SELECT 1 FROM {table} WHERE source_file=? AND source_rel=? LIMIT 1",
            (source_file, source_rel),
        ).fetchone()
    except Exception:
        return source_file
    if not existing:
        return source_file
    try:
        inode = int(file_path.stat().st_ino)
    except Exception:
        inode = 0
    if inode <= 0:
        return source_file
    return f"{source_file}#inode={inode}"


def _sync_file_to_sqlite(
    conn: Optional[sqlite3.Connection],
    table: str,
    project_root: Path,
    file_path: Path,
    start_line: int,
    start_offset_bytes: int,
    dry_run: bool,
    lock_retries: int,
    lock_retry_delay_seconds: float,
    latency_all: Optional[LatencyAccumulator],
    latency_stream: Optional[LatencyAccumulator],
    invalid_log_path: Optional[Path],
    invalid_sample_limit: int,
    run_id: str,
    iter_id: str,
    max_lines_per_file: int = 0,
    max_bytes_per_file: int = 0,
    oversize_payload_bytes: int = 0,
    sqlite_batch_max_bytes: int = 0,
    checkpoint_every_lines: int = 0,
    checkpoint_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    flush_sleep_seconds: float = 0.0,
    host_load_soft_cap: float = 0.0,
    host_load_sleep_seconds: float = 0.0,
) -> Dict[str, Any]:
    inserted = 0
    invalid = 0
    invalid_logged = 0
    oversize_payloads = 0
    ops_write_failures = 0
    cooldown_events = 0
    cooldown_sleep_seconds = 0.0
    last_line_seen = max(int(start_line), 0)
    last_offset_seen = max(int(start_offset_bytes), 0)
    last_checkpoint_line = max(int(start_line), 0)
    source_rel = str(file_path.relative_to(project_root))
    source_stream = _classify_stream(source_rel)
    source_file_identity = _source_file_identity_for_sqlite_insert(conn, table, file_path, source_rel, start_line)
    expected_schema_version = _log_schema_version()
    ops_conn = None
    if not dry_run:
        try:
            ops_conn = ops_data_plane.connect(project_root)
        except Exception:
            ops_conn = None

    rows: List[Tuple[Any, ...]] = []
    rows_payload_bytes = 0
    start_offset_base = max(int(start_offset_bytes), 0)

    def flush_ops() -> None:
        nonlocal ops_conn, ops_write_failures
        if ops_conn is None:
            return
        try:
            ops_conn.commit()
        except Exception:
            ops_write_failures += 1
            try:
                ops_conn.rollback()
            except Exception:
                pass
            try:
                ops_conn.close()
            except Exception:
                pass
            ops_conn = None

    def _record_ops(write_fn: Callable[[], None]) -> None:
        nonlocal ops_conn, ops_write_failures
        if ops_conn is None:
            return
        try:
            write_fn()
        except Exception:
            ops_write_failures += 1
            try:
                ops_conn.rollback()
            except Exception:
                pass
            try:
                ops_conn.close()
            except Exception:
                pass
            ops_conn = None

    def emit_checkpoint(force: bool = False) -> None:
        nonlocal last_checkpoint_line
        if int(last_line_seen) <= int(last_checkpoint_line):
            return
        if not force and max(int(checkpoint_every_lines), 0) > 0:
            if int(last_line_seen) - int(last_checkpoint_line) < int(checkpoint_every_lines):
                return
        if not dry_run:
            if conn is None:
                raise RuntimeError("sqlite connection missing")
            conn.commit()
        if checkpoint_cb is not None:
            checkpoint_cb(
                {
                    "last_line": int(last_line_seen),
                    "last_offset_bytes": int(last_offset_seen),
                    "inserted": int(inserted),
                    "invalid": int(invalid),
                    "invalid_samples_logged": int(invalid_logged),
                    "oversize_payloads": int(oversize_payloads),
                    "ops_write_failures": int(ops_write_failures),
                }
            )
        if ops_conn is not None:
            _record_ops(
                lambda: ops_data_plane.record_watermark(
                    ops_conn,
                    collector_key="jsonl_sql",
                    source_name=table,
                    entity_key=ops_data_plane.normalize_entity_key(project_root, source_rel),
                    watermark_type="line_offset",
                    watermark_value=f"{int(last_line_seen)}:{int(last_offset_seen)}",
                    metadata={
                        "source_rel": source_rel,
                        "table": table,
                        "last_line": int(last_line_seen),
                        "last_offset_bytes": int(last_offset_seen),
                        "inserted": int(inserted),
                        "invalid": int(invalid),
                    },
                    commit=False,
                )
            )
            flush_ops()
        last_checkpoint_line = int(last_line_seen)

    def flush_rows(checkpoint: bool = False) -> None:
        nonlocal inserted, rows, rows_payload_bytes, cooldown_events, cooldown_sleep_seconds
        if not rows:
            return
        if not dry_run:
            if conn is None:
                raise RuntimeError("sqlite connection missing")
            cur = _sqlite_executemany_with_retry(
                conn,
                f"INSERT OR IGNORE INTO {table} (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json, run_id, iter_id, decision_id, parent_decision_id, log_schema_version, source_day_utc, source_stream, source_partition_key, source_broker, source_provider, source_venue, asset_class, routing_lane, source_quality_label) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
                lock_retries=lock_retries,
                lock_retry_delay_seconds=lock_retry_delay_seconds,
            )
            inserted += cur.rowcount if cur.rowcount is not None else 0
        else:
            inserted += len(rows)
        rows = []
        rows_payload_bytes = 0
        if checkpoint:
            emit_checkpoint()
        slept = _ingest_cooldown_sleep(
            base_sleep_seconds=flush_sleep_seconds,
            host_load_soft_cap=host_load_soft_cap,
            host_load_sleep_seconds=host_load_sleep_seconds,
        )
        if slept > 0.0:
            cooldown_events += 1
            cooldown_sleep_seconds += slept

    try:
        for line_no, raw, next_offset in _iter_new_lines(file_path, start_line, start_offset_bytes):
            if int(max_lines_per_file) > 0 and (int(line_no) - max(int(start_line), 0)) > int(max_lines_per_file):
                break
            if int(max_bytes_per_file) > 0:
                processed_lines = int(line_no) - max(int(start_line), 0)
                processed_bytes = max(int(next_offset) - int(start_offset_base), 0)
                if processed_lines > 1 and processed_bytes > int(max_bytes_per_file):
                    break
            last_line_seen = int(line_no)
            last_offset_seen = int(next_offset)
            try:
                obj = json.loads(raw)
                payload = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
            except Exception as exc:
                invalid += 1
                if ops_conn is not None:
                    _record_ops(
                        lambda: ops_data_plane.record_dead_letter(
                            ops_conn,
                            lane="sqlite",
                            source_rel=source_rel,
                            line_no=int(line_no),
                            offset_bytes=int(next_offset),
                            error_class=exc.__class__.__name__,
                            error_message=str(exc),
                            raw_payload=raw,
                            run_id=run_id,
                            iter_id=iter_id,
                            metadata={"table": table, "file_path": str(file_path)},
                            commit=False,
                        )
                    )
                if invalid_logged < max(int(invalid_sample_limit), 0):
                    _log_invalid_line(
                        invalid_log_path=invalid_log_path,
                        mode="sqlite",
                        source_rel=source_rel,
                        line_no=line_no,
                        raw=raw,
                        error=exc,
                        run_id=run_id,
                        iter_id=iter_id,
                    )
                    invalid_logged += 1
                continue

            payload_bytes = len(payload.encode("utf-8"))
            if int(oversize_payload_bytes) > 0 and payload_bytes > int(oversize_payload_bytes):
                invalid += 1
                oversize_payloads += 1
                exc = ValueError(
                    f"payload_size_bytes {payload_bytes} exceeds oversize_payload_bytes {int(oversize_payload_bytes)}"
                )
                if ops_conn is not None:
                    _record_ops(
                        lambda: ops_data_plane.record_dead_letter(
                            ops_conn,
                            lane="sqlite",
                            source_rel=source_rel,
                            line_no=int(line_no),
                            offset_bytes=int(next_offset),
                            error_class="OversizePayload",
                            error_message=str(exc),
                            raw_payload=raw[:512],
                            run_id=run_id,
                            iter_id=iter_id,
                            metadata={
                                "table": table,
                                "file_path": str(file_path),
                                "payload_size_bytes": int(payload_bytes),
                                "oversize_payload_bytes": int(oversize_payload_bytes),
                            },
                            commit=False,
                        )
                    )
                if invalid_logged < max(int(invalid_sample_limit), 0):
                    _log_invalid_line(
                        invalid_log_path=invalid_log_path,
                        mode="sqlite",
                        source_rel=source_rel,
                        line_no=line_no,
                        raw=raw,
                        error=exc,
                        run_id=run_id,
                        iter_id=iter_id,
                    )
                    invalid_logged += 1
                emit_checkpoint()
                continue

            event_ts = _extract_event_ts_utc(obj)
            if event_ts is not None:
                latency_s = max(time.time() - event_ts.timestamp(), 0.0)
                if latency_all is not None:
                    latency_all.add(latency_s)
                if latency_stream is not None:
                    latency_stream.add(latency_s)

            raw_schema = obj.get("log_schema_version")
            metadata_obj = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            if raw_schema is None:
                raw_schema = metadata_obj.get("log_schema_version")
            try:
                observed_schema_version = max(int(raw_schema), 0)
            except Exception:
                observed_schema_version = 0

            run_id_row, iter_id_row, decision_id, parent_decision_id, schema_version = _extract_correlation_fields(obj)
            (
                source_broker,
                source_provider,
                source_venue,
                asset_class,
                routing_lane,
                source_quality_label,
            ) = _extract_route_fields(obj, source_rel=source_rel)
            source_day_utc = _event_day_utc(event_ts, fallback_ts=datetime.now(timezone.utc))
            source_partition_key = f"{source_day_utc}:{source_stream}"
            if ops_conn is not None and observed_schema_version != expected_schema_version:
                _record_ops(
                    lambda: ops_data_plane.record_schema_drift(
                        ops_conn,
                        lane="sqlite",
                        source_rel=source_rel,
                        line_no=int(line_no),
                        observed_schema_version=int(observed_schema_version),
                        expected_schema_version=int(expected_schema_version),
                        drift_kind="missing_log_schema_version" if observed_schema_version <= 0 else "schema_version_mismatch",
                        payload_json=payload,
                        run_id=run_id_row or run_id,
                        iter_id=iter_id_row or iter_id,
                        metadata={"table": table, "file_path": str(file_path)},
                        commit=False,
                    )
                )
            sha1 = hashlib.sha1(payload.encode("utf-8")).hexdigest()
            rows.append(
                (
                    source_file_identity,
                    source_rel,
                    line_no,
                    _now_utc(),
                    sha1,
                    payload,
                    run_id_row,
                    iter_id_row,
                    decision_id,
                    parent_decision_id,
                    schema_version,
                    source_day_utc,
                    source_stream,
                    source_partition_key,
                    source_broker,
                    source_provider,
                    source_venue,
                    asset_class,
                    routing_lane,
                    source_quality_label,
                )
            )
            rows_payload_bytes += int(payload_bytes)

            if len(rows) >= 1000 or (
                int(sqlite_batch_max_bytes) > 0 and int(rows_payload_bytes) >= int(sqlite_batch_max_bytes)
            ):
                flush_rows(checkpoint=True)

        flush_rows()

        emit_checkpoint(force=True)
    finally:
        if ops_conn is not None:
            flush_ops()
            ops_conn.close()

    return {
        "inserted": int(inserted),
        "invalid": int(invalid),
        "invalid_samples_logged": int(invalid_logged),
        "oversize_payloads": int(oversize_payloads),
        "ops_write_failures": int(ops_write_failures),
        "cooldown_events": int(cooldown_events),
        "cooldown_sleep_seconds": round(float(cooldown_sleep_seconds), 3),
        "last_line": int(last_line_seen),
        "last_offset_bytes": int(last_offset_seen),
    }


def _mysql_escape(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _mysql_exec(mysql_bin: str, host: str, port: int, user: str, password: str, database: str, sql: str) -> None:
    env = os.environ.copy()
    if password:
        env["MYSQL_PWD"] = password
    cmd = [mysql_bin, "-h", host, "-P", str(port), "-u", user, database, "-e", sql]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "mysql command failed").strip())


def _mysql_exec_allow_duplicate(
    mysql_bin: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    sql: str,
    *,
    duplicate_markers: Tuple[str, ...],
) -> None:
    try:
        _mysql_exec(mysql_bin, host, port, user, password, database, sql)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if any(marker in msg for marker in duplicate_markers):
            return
        raise


def _ensure_mysql_schema(
    mysql_bin: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table: str,
) -> None:
    _mysql_exec(
        mysql_bin,
        host,
        port,
        user,
        password,
        database,
        f"""
    CREATE TABLE IF NOT EXISTS {table} (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      source_file TEXT NOT NULL,
      source_rel VARCHAR(1024) NOT NULL,
      line_no BIGINT NOT NULL,
      ingested_at VARCHAR(64) NOT NULL,
      payload_sha1 VARCHAR(40) NOT NULL,
      payload_json LONGTEXT NOT NULL,
      run_id VARCHAR(192) NULL,
      iter_id VARCHAR(192) NULL,
      decision_id VARCHAR(192) NULL,
      parent_decision_id VARCHAR(192) NULL,
      log_schema_version INT NULL,
      UNIQUE KEY uniq_source_line (source_rel(255), line_no),
      KEY idx_ingested_at (ingested_at),
      KEY idx_run_id (run_id),
      KEY idx_iter_id (iter_id),
      KEY idx_decision_id (decision_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    )

    for sql in [
        f"ALTER TABLE {table} ADD COLUMN run_id VARCHAR(192) NULL",
        f"ALTER TABLE {table} ADD COLUMN iter_id VARCHAR(192) NULL",
        f"ALTER TABLE {table} ADD COLUMN decision_id VARCHAR(192) NULL",
        f"ALTER TABLE {table} ADD COLUMN parent_decision_id VARCHAR(192) NULL",
        f"ALTER TABLE {table} ADD COLUMN log_schema_version INT NULL",
    ]:
        _mysql_exec_allow_duplicate(
            mysql_bin,
            host,
            port,
            user,
            password,
            database,
            sql,
            duplicate_markers=("duplicate column name", "error 1060"),
        )

    for sql in [
        f"ALTER TABLE {table} ADD INDEX idx_run_id (run_id)",
        f"ALTER TABLE {table} ADD INDEX idx_iter_id (iter_id)",
        f"ALTER TABLE {table} ADD INDEX idx_decision_id (decision_id)",
    ]:
        _mysql_exec_allow_duplicate(
            mysql_bin,
            host,
            port,
            user,
            password,
            database,
            sql,
            duplicate_markers=("duplicate key name", "error 1061"),
        )


def _sync_file_to_mysql(
    mysql_bin: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table: str,
    project_root: Path,
    file_path: Path,
    start_line: int,
    start_offset_bytes: int,
    batch_size: int,
    dry_run: bool,
    latency_all: Optional[LatencyAccumulator],
    latency_stream: Optional[LatencyAccumulator],
    invalid_log_path: Optional[Path],
    invalid_sample_limit: int,
    run_id: str,
    iter_id: str,
    max_lines_per_file: int = 0,
    max_bytes_per_file: int = 0,
    oversize_payload_bytes: int = 0,
) -> Dict[str, Any]:
    inserted = 0
    invalid = 0
    invalid_logged = 0
    oversize_payloads = 0
    last_line_seen = max(int(start_line), 0)
    last_offset_seen = max(int(start_offset_bytes), 0)
    start_offset_base = max(int(start_offset_bytes), 0)
    vals: List[str] = []
    source_rel = str(file_path.relative_to(project_root))

    def flush() -> None:
        nonlocal inserted, vals
        if not vals:
            return
        if dry_run:
            inserted += len(vals)
            vals = []
            return
        sql = (
            f"INSERT IGNORE INTO {table} "
            "(source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json, run_id, iter_id, decision_id, parent_decision_id, log_schema_version) VALUES "
            + ",".join(vals)
            + ";"
        )
        _mysql_exec(mysql_bin, host, port, user, password, database, sql)
        inserted += len(vals)
        vals = []

    for line_no, raw, next_offset in _iter_new_lines(file_path, start_line, start_offset_bytes):
        if int(max_lines_per_file) > 0 and (int(line_no) - max(int(start_line), 0)) > int(max_lines_per_file):
            break
        if int(max_bytes_per_file) > 0:
            processed_lines = int(line_no) - max(int(start_line), 0)
            processed_bytes = max(int(next_offset) - int(start_offset_base), 0)
            if processed_lines > 1 and processed_bytes > int(max_bytes_per_file):
                break
        last_line_seen = int(line_no)
        last_offset_seen = int(next_offset)
        try:
            obj = json.loads(raw)
            payload = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
        except Exception as exc:
            invalid += 1
            if invalid_logged < max(int(invalid_sample_limit), 0):
                _log_invalid_line(
                    invalid_log_path=invalid_log_path,
                    mode="mysql",
                    source_rel=source_rel,
                    line_no=line_no,
                    raw=raw,
                    error=exc,
                    run_id=run_id,
                    iter_id=iter_id,
                )
                invalid_logged += 1
            continue

        payload_bytes = len(payload.encode("utf-8"))
        if int(oversize_payload_bytes) > 0 and payload_bytes > int(oversize_payload_bytes):
            invalid += 1
            oversize_payloads += 1
            exc = ValueError(
                f"payload_size_bytes {payload_bytes} exceeds oversize_payload_bytes {int(oversize_payload_bytes)}"
            )
            if invalid_logged < max(int(invalid_sample_limit), 0):
                _log_invalid_line(
                    invalid_log_path=invalid_log_path,
                    mode="mysql",
                    source_rel=source_rel,
                    line_no=line_no,
                    raw=raw,
                    error=exc,
                    run_id=run_id,
                    iter_id=iter_id,
                )
                invalid_logged += 1
            continue

        event_ts = _extract_event_ts_utc(obj)
        if event_ts is not None:
            latency_s = max(time.time() - event_ts.timestamp(), 0.0)
            if latency_all is not None:
                latency_all.add(latency_s)
            if latency_stream is not None:
                latency_stream.add(latency_s)

        run_id_row, iter_id_row, decision_id, parent_decision_id, schema_version = _extract_correlation_fields(obj)

        source_file = _mysql_escape(str(file_path))
        source_rel_esc = _mysql_escape(source_rel)
        ingested_at = _mysql_escape(_now_utc())
        sha1 = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        payload_esc = _mysql_escape(payload)

        vals.append(
            "("
            f"'{source_file}','{source_rel_esc}',{line_no},'{ingested_at}','{sha1}','{payload_esc}',"
            f"'{_mysql_escape(run_id_row)}','{_mysql_escape(iter_id_row)}','{_mysql_escape(decision_id)}','{_mysql_escape(parent_decision_id)}',"
            f"{int(schema_version)}"
            ")"
        )

        if len(vals) >= batch_size:
            flush()

    flush()

    return {
        "inserted": int(inserted),
        "invalid": int(invalid),
        "invalid_samples_logged": int(invalid_logged),
        "oversize_payloads": int(oversize_payloads),
        "last_line": int(last_line_seen),
        "last_offset_bytes": int(last_offset_seen),
    }


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _health_counter(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except Exception:
        return 0


def _parse_iso_utc(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_health_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _health_file_shard_name(path: Path) -> str:
    name = path.name
    prefix = "jsonl_sql_ingestion_health_"
    suffix = "_latest.json"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)].strip().lower()
    return ""


def _stale_decision_catch_up_requested(shard_name: str, path_contains: Optional[List[str]] = None) -> bool:
    if not _env_flag("SQL_LINK_SERVICE_STALE_DECISION_SOURCE_CATCH_UP", False):
        return False
    if str(shard_name or "").strip() not in {"trading", "aggressive_trading", "crypto_trading"}:
        return False
    return any("decision_" in str(token) or "trade_decisions_" in str(token) for token in (path_contains or []))


def _fresh_idle_health_fast_path_allowed(
    health_file_path: Path,
    *,
    path_contains: Optional[List[str]] = None,
    source_files: Optional[List[Path]] = None,
    project_root: Optional[Path] = None,
    sqlite_state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    if not _env_flag("SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS", False):
        return False, {"reason": "disabled"}
    shard_name = _health_file_shard_name(health_file_path)
    sentinel_shards = {"health_fast", "writer_progress"}
    if shard_name in sentinel_shards and not _env_flag("SQL_LINK_SERVICE_SKIP_IDLE_SENTINELS", False):
        return False, {"reason": "sentinel_shard"}
    if _stale_decision_catch_up_requested(shard_name, path_contains):
        return False, {"reason": "stale_decision_catch_up", "shard": shard_name}
    health = _load_health_payload(health_file_path)
    if not health:
        return False, {"reason": "missing_health"}
    status = str(health.get("overall_status") or health.get("status") or "").strip().lower()
    if status in {"error", "failed", "blocked"}:
        return False, {"reason": "last_health_not_clean", "status": status}
    timestamp = _parse_iso_utc(health.get("timestamp_utc"))
    if timestamp is None:
        return False, {"reason": "missing_timestamp"}
    age_seconds = max((datetime.now(timezone.utc) - timestamp).total_seconds(), 0.0)
    max_age_seconds = max(_env_float("SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS", 90.0), 1.0)
    if age_seconds > max_age_seconds:
        return False, {"reason": "health_stale", "age_seconds": round(age_seconds, 3), "max_age_seconds": max_age_seconds}
    sqlite_bucket = health.get("sqlite") if isinstance(health.get("sqlite"), dict) else {}
    sqlite_json_bucket = health.get("sqlite_json_files") if isinstance(health.get("sqlite_json_files"), dict) else {}
    dirty_counts = {
        "invalid": _health_counter(sqlite_bucket.get("invalid")) + _health_counter(health.get("invalid_lines")),
        "oversize_payloads": _health_counter(sqlite_bucket.get("oversize_payloads")),
        "ops_write_failures": _health_counter(sqlite_bucket.get("ops_write_failures")),
        "json_invalid": _health_counter(sqlite_json_bucket.get("invalid")),
    }
    if any(count > 0 for count in dirty_counts.values()):
        return False, {"reason": "last_health_has_ingestion_errors", "shard": shard_name, "counts": dirty_counts}
    pending_lines = max(_health_counter(sqlite_bucket.get("pending_lines")), _health_counter(health.get("pending_lines")))
    pending_json_files = max(
        _health_counter(sqlite_json_bucket.get("pending_files")),
        _health_counter(sqlite_json_bucket.get("pending")),
        _health_counter(health.get("pending_json_files")),
    )
    if pending_lines > 0 or pending_json_files > 0:
        return False, {
            "reason": "pending_work_present",
            "pending_lines": int(pending_lines),
            "pending_json_files": int(pending_json_files),
        }
    if source_files is not None:
        root = project_root.resolve() if project_root is not None else None
        state_rows = sqlite_state if isinstance(sqlite_state, dict) else {}
        for source_path in source_files:
            try:
                resolved = source_path.resolve()
                source_stat = resolved.stat()
            except Exception:
                return False, {
                    "reason": "source_state_not_idle",
                    "source": str(source_path),
                    "source_reason": "source_stat_failed",
                }
            state_keys = [str(resolved)]
            if root is not None:
                try:
                    state_keys.insert(0, resolved.relative_to(root).as_posix())
                except Exception:
                    pass
            state_row = next(
                (state_rows.get(key) for key in state_keys if isinstance(state_rows.get(key), dict)),
                None,
            )
            if not isinstance(state_row, dict):
                return False, {
                    "reason": "source_state_not_idle",
                    "source": str(resolved),
                    "source_reason": "source_untracked",
                }
            last_offset = _health_counter(state_row.get("last_offset_bytes"))
            state_inode = _health_counter(state_row.get("file_inode"))
            if state_inode > 0 and state_inode != int(source_stat.st_ino):
                return False, {
                    "reason": "source_state_not_idle",
                    "source": str(resolved),
                    "source_reason": "source_inode_changed",
                }
            if int(source_stat.st_size) > last_offset:
                return False, {
                    "reason": "source_state_not_idle",
                    "source": str(resolved),
                    "source_reason": "pending_source_bytes",
                    "pending_bytes": int(source_stat.st_size) - last_offset,
                }
            if last_offset > int(source_stat.st_size):
                return False, {
                    "reason": "source_state_not_idle",
                    "source": str(resolved),
                    "source_reason": "source_truncated",
                }
            if float(source_stat.st_mtime) > timestamp.timestamp():
                return False, {
                    "reason": "source_state_not_idle",
                    "source": str(resolved),
                    "source_reason": "source_changed_after_health",
                }
    return True, {
        "reason": "fresh_idle_health",
        "shard": shard_name,
        "age_seconds": round(age_seconds, 3),
        "max_age_seconds": max_age_seconds,
        "health_file": str(health_file_path),
    }


def _journal_event_allowed(payload: Dict[str, Any]) -> bool:
    event = str(payload.get("event") or "").strip()
    if event == "file_failed" and _env_flag("INGEST_JOURNAL_ERRORS_ALWAYS", True):
        return True
    if not _env_flag("INGEST_JOURNAL_ENABLED", True):
        return False
    if event == "file_start" and not _env_flag("INGEST_JOURNAL_FILE_START_ENABLED", True):
        return False
    if event == "file_checkpoint" and not _env_flag("INGEST_JOURNAL_CHECKPOINT_ENABLED", True):
        return False
    if (
        event in {"file_checkpoint", "file_complete"}
        and not _env_flag("INGEST_JOURNAL_ZERO_PENDING_ENABLED", True)
        and int(payload.get("pending_lines") or 0) <= 0
    ):
        return False
    return True


def _journal_path_allowed(path: Path) -> bool:
    normalized = str(path).replace("\\", "/")
    if "/governance/events/" in normalized and not _env_flag("INGEST_JOURNAL_DAILY_ENABLED", True):
        return False
    return True


def _journal_event(paths: List[Path], payload: Dict[str, Any]) -> None:
    if not _journal_event_allowed(payload):
        return
    for path in paths:
        if _journal_path_allowed(path):
            _append_jsonl(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Link all project JSONL files to SQL (SQLite/MySQL).")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--mode", choices=["sqlite", "mysql", "both"], default="both")
    parser.add_argument("--state-file", default=None, help="Path to incremental ingest state JSON.")

    parser.add_argument("--sqlite-db", default=None, help="SQLite database file path.")
    parser.add_argument("--sqlite-table", default="jsonl_records")
    parser.add_argument("--sqlite-json-table", default="json_file_records")
    parser.add_argument("--sqlite-timeout-seconds", type=float, default=float(os.getenv("SQLITE_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--sqlite-lock-retries", type=int, default=int(os.getenv("SQLITE_LOCK_RETRIES", "8")))
    parser.add_argument(
        "--sqlite-lock-retry-delay-seconds",
        type=float,
        default=float(os.getenv("SQLITE_LOCK_RETRY_DELAY_SECONDS", "0.25")),
    )
    parser.add_argument(
        "--sqlite-state-checkpoint-lines",
        type=int,
        default=int(os.getenv("SQLITE_STATE_CHECKPOINT_LINES", "10000")),
    )

    parser.add_argument("--mysql-bin", default=os.getenv("MYSQL_BIN", "/opt/homebrew/bin/mysql"))
    parser.add_argument("--mysql-host", default=os.getenv("MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--mysql-port", type=int, default=int(os.getenv("MYSQL_PORT", "3306")))
    parser.add_argument("--mysql-user", default=os.getenv("MYSQL_USER", "root"))
    parser.add_argument("--mysql-password", default=os.getenv("MYSQL_PASSWORD", ""))
    parser.add_argument("--mysql-database", default=os.getenv("MYSQL_DATABASE", "schwab_trading"))
    parser.add_argument("--mysql-table", default=os.getenv("MYSQL_TABLE", "jsonl_records"))
    parser.add_argument("--mysql-batch-size", type=int, default=int(os.getenv("MYSQL_BATCH_SIZE", "200")))

    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--max-lines-per-file",
        type=int,
        default=int(os.getenv("INGEST_MAX_LINES_PER_FILE", "0")),
    )
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=int(os.getenv("INGEST_MAX_BYTES_PER_FILE", "0")),
        help="Maximum bytes to process from each file per run; 0 disables the cap.",
    )
    parser.add_argument(
        "--oversize-payload-bytes",
        type=int,
        default=int(os.getenv("INGEST_OVERSIZE_PAYLOAD_BYTES", "0")),
        help="Dead-letter JSONL payloads larger than this byte size; 0 disables the cap.",
    )
    parser.add_argument(
        "--sqlite-batch-max-bytes",
        type=int,
        default=int(os.getenv("SQLITE_BATCH_MAX_BYTES", "0")),
        help="Flush SQLite batches once accumulated payload bytes exceed this size; 0 keeps row-count batching only.",
    )
    parser.add_argument(
        "--max-deferred-files",
        type=int,
        default=int(os.getenv("INGEST_MAX_DEFERRED_FILES", "2")),
    )
    parser.add_argument("--top-pending-files", type=int, default=int(os.getenv("INGEST_TOP_PENDING_FILES", "10")))
    parser.add_argument("--invalid-sample-limit", type=int, default=int(os.getenv("INGEST_INVALID_SAMPLE_LIMIT", "25")))
    parser.add_argument("--invalid-log-file", default=os.getenv("INGEST_INVALID_LOG_FILE", ""))
    parser.add_argument("--journal-file", default=os.getenv("INGEST_JOURNAL_FILE", ""))
    parser.add_argument("--journal-events-file", default=os.getenv("INGEST_JOURNAL_EVENTS_FILE", ""))
    parser.add_argument("--health-file", default=os.getenv("INGEST_HEALTH_FILE", ""))
    parser.add_argument("--include-streams", default=os.getenv("INGEST_INCLUDE_STREAMS", ""))
    parser.add_argument("--exclude-streams", default=os.getenv("INGEST_EXCLUDE_STREAMS", ""))
    parser.add_argument("--path-contains", default=os.getenv("INGEST_PATH_CONTAINS", ""))
    parser.add_argument("--path-not-contains", default=os.getenv("INGEST_PATH_NOT_CONTAINS", ""))
    parser.add_argument(
        "--ingest-flush-sleep-seconds",
        type=float,
        default=float(os.getenv("INGEST_FLUSH_SLEEP_SECONDS", "0")),
        help="Sleep after SQLite flushes to keep JSONL catch-up from monopolizing CPU.",
    )
    parser.add_argument(
        "--ingest-file-sleep-seconds",
        type=float,
        default=float(os.getenv("INGEST_FILE_SLEEP_SECONDS", "0")),
        help="Sleep after each processed JSONL file.",
    )
    parser.add_argument(
        "--ingest-host-load-soft-cap",
        type=float,
        default=float(os.getenv("INGEST_HOST_LOAD_SOFT_CAP", "0")),
        help="When the 1m host load is at or above this cap, apply the host-load ingestion sleep.",
    )
    parser.add_argument(
        "--ingest-host-load-sleep-seconds",
        type=float,
        default=float(os.getenv("INGEST_HOST_LOAD_SLEEP_SECONDS", "0")),
        help="Minimum sleep applied after flushes/files while host load is above the soft cap.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-json-files", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"Project root missing: {project_root}")
        return 2
    maintenance_hold = maintenance_hold_snapshot(project_root)
    if (
        bool(maintenance_hold.get("active", False))
        and not maintenance_hold_token_authorized(maintenance_hold)
        and not bool(args.dry_run)
    ):
        print(
            "link_jsonl_to_sql guarded_hold=runtime_maintenance_hold_active "
            f"reason={maintenance_hold.get('reason', 'runtime_maintenance')}"
        )
        return 75

    include_streams = _parse_csv_values(args.include_streams)
    exclude_streams = _parse_csv_values(args.exclude_streams)
    path_contains = _parse_csv_values(args.path_contains)
    path_not_contains = _parse_csv_values(args.path_not_contains)

    files = discover_jsonl_files(project_root)
    files = [
        fp
        for fp in files
        if _matches_rel_filters(
            source_rel=str(fp.relative_to(project_root)),
            stream=_classify_stream(str(fp.relative_to(project_root))),
            include_streams=include_streams,
            exclude_streams=exclude_streams,
            path_contains=path_contains,
            path_not_contains=path_not_contains,
        )
    ]
    json_files = [] if args.skip_json_files else discover_json_files(project_root)
    if not args.skip_json_files:
        json_files = [
            fp
            for fp in json_files
            if _matches_rel_filters(
                source_rel=str(fp.relative_to(project_root)),
                stream=_json_file_stream(str(fp.relative_to(project_root))),
                include_streams=include_streams,
                exclude_streams=exclude_streams,
                path_contains=path_contains,
                path_not_contains=path_not_contains,
            )
        ]
    state_path = (
        Path(args.state_file).resolve()
        if args.state_file
        else (project_root / "governance" / "jsonl_sql_link_state.json")
    )
    state = _load_state(state_path)
    files = _prioritize_jsonl_files_by_pending_bytes(
        files,
        project_root=project_root,
        sqlite_state=state.get("sqlite", {}) if isinstance(state, dict) else {},
    )
    idle_check_files = list(files)

    if args.max_files > 0:
        files = _limit_prioritized_jsonl_files(
            files,
            project_root=project_root,
            max_files=args.max_files,
            max_deferred_files=max(args.max_deferred_files, 0),
        )
        json_files = json_files[: args.max_files]

    print(f"Discovered JSONL files: {len(files)}")
    if args.skip_json_files:
        print("Discovered JSON files: 0 (skip-json-files)")
    else:
        print(f"Discovered JSON files: {len(json_files)}")

    day_utc = datetime.now(timezone.utc).strftime("%Y%m%d")
    default_invalid_log = project_root / "governance" / "events" / f"jsonl_ingestion_invalid_{day_utc}.jsonl"
    invalid_log_path = Path(args.invalid_log_file).resolve() if args.invalid_log_file else default_invalid_log

    default_journal_latest = project_root / "governance" / "health" / "jsonl_ingest_batch_journal_latest.jsonl"
    default_journal_daily = project_root / "governance" / "events" / f"jsonl_ingest_batches_{day_utc}.jsonl"
    journal_paths = [
        Path(args.journal_file).resolve() if args.journal_file else default_journal_latest,
        Path(args.journal_events_file).resolve() if args.journal_events_file else default_journal_daily,
    ]
    health_file_path = (
        Path(args.health_file).resolve()
        if args.health_file
        else (project_root / "governance" / "health" / "jsonl_sql_ingestion_health_latest.json")
    )
    fast_path_allowed, fast_path = _fresh_idle_health_fast_path_allowed(
        health_file_path,
        path_contains=path_contains,
        source_files=idle_check_files,
        project_root=project_root,
        sqlite_state=state.get("sqlite", {}) if isinstance(state, dict) else {},
    )
    if fast_path_allowed:
        print(
            "Fresh idle health fast-path skip: "
            f"shard={fast_path.get('shard', '')} "
            f"age_seconds={fast_path.get('age_seconds', '')} "
            f"health_file={health_file_path}"
        )
        return 0

    run_id = str(os.getenv("CORRELATION_RUN_ID", "") or "").strip()
    iter_id = str(os.getenv("CORRELATION_ITER_ID", "") or "").strip()
    ingest_run_id = run_id or f"ingest-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"

    sqlite_conn: Optional[sqlite3.Connection] = None
    if args.mode in {"sqlite", "both"}:
        sqlite_db = Path(args.sqlite_db).resolve() if args.sqlite_db else (project_root / "data" / "jsonl_link.sqlite3")
        if not args.dry_run:
            sqlite_conn = connect_sqlite(
                sqlite_db,
                project_root=project_root,
                timeout_seconds=max(float(args.sqlite_timeout_seconds), 1.0),
            )
            _ensure_sqlite_schema(sqlite_conn, args.sqlite_table)
            _ensure_sqlite_json_file_schema(sqlite_conn, args.sqlite_json_table)
        print(f"SQLite target: {sqlite_db} table={args.sqlite_table}")

    if args.mode in {"mysql", "both"}:
        if not Path(args.mysql_bin).exists():
            print(f"MySQL CLI not found: {args.mysql_bin}")
            return 2
        if not args.dry_run:
            _ensure_mysql_schema(
                args.mysql_bin,
                args.mysql_host,
                args.mysql_port,
                args.mysql_user,
                args.mysql_password,
                args.mysql_database,
                args.mysql_table,
            )
        print(
            f"MySQL target: host={args.mysql_host}:{args.mysql_port} db={args.mysql_database} table={args.mysql_table} user={args.mysql_user}"
        )

    total_inserted = {"sqlite": 0, "mysql": 0}
    total_invalid = {"sqlite": 0, "mysql": 0}
    total_invalid_samples = {"sqlite": 0, "mysql": 0}
    json_file_metrics = {
        "sqlite": {
            "inserted": 0,
            "invalid": 0,
            "skipped_unchanged": 0,
            "bytes": 0,
        }
    }
    lag_metrics = {
        "sqlite": {
            "pending_lines": 0,
            "oldest_uningested_age_seconds": 0.0,
            "files_with_pending": 0,
            "top_pending_files": [],
            "oversize_payloads": 0,
        },
            "mysql": {
                "pending_lines": 0,
                "oldest_uningested_age_seconds": 0.0,
                "files_with_pending": 0,
                "top_pending_files": [],
                "oversize_payloads": 0,
            },
    }
    cooldown_metrics = {
        "events": 0,
        "sleep_seconds": 0.0,
    }
    latency_metrics: Dict[str, Dict[str, Any]] = {
        "sqlite": {"all": LatencyAccumulator(), "by_stream": {}},
        "mysql": {"all": LatencyAccumulator(), "by_stream": {}},
    }

    try:
        for fp in files:
            rel = str(fp.relative_to(project_root))
            stream = _classify_stream(rel)
            try:
                st = fp.stat()
                mtime = float(st.st_mtime)
            except FileNotFoundError:
                print(f"Skipping vanished file before sync: {rel}")
                continue

            print(f"Syncing: {rel}")
            total_lines = _count_lines(fp)

            if args.mode in {"sqlite", "both"}:
                progress = state["sqlite"].get(rel, {"last_line": 0, "mtime": 0.0})
                start_line, start_offset, reset_reason = _derive_start_cursor(progress, st)
                start_evt = {
                    "timestamp_utc": _now_utc(),
                    "event": "file_start",
                    "ingest_run_id": ingest_run_id,
                    "mode": "sqlite",
                    "source_rel": rel,
                    "stream": stream,
                    "start_line": int(start_line),
                    "start_offset_bytes": int(start_offset),
                    "reset_reason": str(reset_reason),
                }
                _journal_event(journal_paths, start_evt)

                def sqlite_checkpoint(checkpoint: Dict[str, Any]) -> None:
                    try:
                        checkpoint_st = fp.stat()
                    except Exception:
                        checkpoint_st = st
                    state["sqlite"][rel] = {
                        "last_line": int(checkpoint["last_line"]),
                        "last_offset_bytes": int(checkpoint["last_offset_bytes"]),
                        "mtime": float(checkpoint_st.st_mtime),
                        "file_inode": int(checkpoint_st.st_ino),
                        "file_size_bytes": int(checkpoint_st.st_size),
                    }
                    _save_state(state_path, state)
                    pending_lines = max(int(total_lines) - int(checkpoint["last_line"]), 0)
                    _journal_event(
                        journal_paths,
                        {
                            "timestamp_utc": _now_utc(),
                            "event": "file_checkpoint",
                            "ingest_run_id": ingest_run_id,
                            "mode": "sqlite",
                            "source_rel": rel,
                            "stream": stream,
                            "last_line": int(checkpoint["last_line"]),
                            "last_offset_bytes": int(checkpoint["last_offset_bytes"]),
                            "inserted": int(checkpoint["inserted"]),
                            "invalid": int(checkpoint["invalid"]),
                            "invalid_samples_logged": int(checkpoint["invalid_samples_logged"]),
                            "oversize_payloads": int(checkpoint.get("oversize_payloads", 0) or 0),
                            "ops_write_failures": int(checkpoint.get("ops_write_failures", 0) or 0),
                            "pending_lines": int(pending_lines),
                        },
                    )

                lat_all, lat_stream = _ensure_latency_bucket(latency_metrics, "sqlite", stream)
                started_ts = time.time()
                try:
                    result = _sync_file_to_sqlite(
                        sqlite_conn,
                        args.sqlite_table,
                        project_root,
                        fp,
                        start_line,
                        start_offset,
                        args.dry_run,
                        lock_retries=max(args.sqlite_lock_retries, 0),
                        lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                        latency_all=lat_all,
                        latency_stream=lat_stream,
                        invalid_log_path=invalid_log_path,
                        invalid_sample_limit=max(args.invalid_sample_limit, 0),
                        run_id=run_id,
                        iter_id=iter_id,
                        max_lines_per_file=max(int(args.max_lines_per_file), 0),
                        max_bytes_per_file=max(int(args.max_bytes_per_file), 0),
                        oversize_payload_bytes=max(int(args.oversize_payload_bytes), 0),
                        sqlite_batch_max_bytes=max(int(args.sqlite_batch_max_bytes), 0),
                        checkpoint_every_lines=max(int(args.sqlite_state_checkpoint_lines), 0),
                        checkpoint_cb=sqlite_checkpoint,
                        flush_sleep_seconds=max(float(args.ingest_flush_sleep_seconds), 0.0),
                        host_load_soft_cap=max(float(args.ingest_host_load_soft_cap), 0.0),
                        host_load_sleep_seconds=max(float(args.ingest_host_load_sleep_seconds), 0.0),
                    )
                except FileNotFoundError:
                    _journal_event(
                        journal_paths,
                        {
                            "timestamp_utc": _now_utc(),
                            "event": "file_failed",
                            "ingest_run_id": ingest_run_id,
                            "mode": "sqlite",
                            "source_rel": rel,
                            "stream": stream,
                            "error": "file_vanished_during_sync",
                        },
                    )
                    print(f"  sqlite skipped vanished file during sync: {rel}")
                    continue
                except Exception as exc:
                    _journal_event(
                        journal_paths,
                        {
                            "timestamp_utc": _now_utc(),
                            "event": "file_failed",
                            "ingest_run_id": ingest_run_id,
                            "mode": "sqlite",
                            "source_rel": rel,
                            "stream": stream,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise

                if not args.dry_run and sqlite_conn is not None:
                    sqlite_conn.commit()

                total_inserted["sqlite"] += int(result["inserted"])
                total_invalid["sqlite"] += int(result["invalid"])
                total_invalid_samples["sqlite"] += int(result["invalid_samples_logged"])
                lag_metrics["sqlite"]["ops_write_failures"] = int(
                    lag_metrics["sqlite"].get("ops_write_failures", 0) or 0
                ) + int(result.get("ops_write_failures", 0) or 0)
                lag_metrics["sqlite"]["oversize_payloads"] = int(
                    lag_metrics["sqlite"].get("oversize_payloads", 0) or 0
                ) + int(result.get("oversize_payloads", 0) or 0)
                cooldown_metrics["events"] += int(result.get("cooldown_events", 0) or 0)
                cooldown_metrics["sleep_seconds"] += float(result.get("cooldown_sleep_seconds", 0.0) or 0.0)

                try:
                    post_st = fp.stat()
                except Exception:
                    post_st = st

                state["sqlite"][rel] = {
                    "last_line": int(result["last_line"]),
                    "last_offset_bytes": int(result["last_offset_bytes"]),
                    "mtime": float(post_st.st_mtime),
                    "file_inode": int(post_st.st_ino),
                    "file_size_bytes": int(post_st.st_size),
                }

                pending_lines = max(int(total_lines) - int(result["last_line"]), 0)
                oldest_age = max(time.time() - mtime, 0.0) if pending_lines > 0 else 0.0
                lag_metrics["sqlite"]["pending_lines"] += pending_lines
                lag_metrics["sqlite"]["oldest_uningested_age_seconds"] = max(
                    float(lag_metrics["sqlite"].get("oldest_uningested_age_seconds", 0.0) or 0.0),
                    float(oldest_age),
                )
                if pending_lines > 0:
                    lag_metrics["sqlite"]["files_with_pending"] += 1
                    _record_top_pending(
                        lag_metrics["sqlite"]["top_pending_files"],
                        source_rel=rel,
                        pending_lines=pending_lines,
                        oldest_age_seconds=oldest_age,
                        total_lines=total_lines,
                        last_line=int(result["last_line"]),
                        top_n=max(int(args.top_pending_files), 1),
                    )

                _journal_event(
                    journal_paths,
                    {
                        "timestamp_utc": _now_utc(),
                        "event": "file_complete",
                        "ingest_run_id": ingest_run_id,
                        "mode": "sqlite",
                        "source_rel": rel,
                        "stream": stream,
                        "inserted": int(result["inserted"]),
                        "invalid": int(result["invalid"]),
                        "invalid_samples_logged": int(result["invalid_samples_logged"]),
                        "oversize_payloads": int(result.get("oversize_payloads", 0) or 0),
                        "ops_write_failures": int(result.get("ops_write_failures", 0) or 0),
                        "last_line": int(result["last_line"]),
                        "last_offset_bytes": int(result["last_offset_bytes"]),
                        "pending_lines": int(pending_lines),
                        "duration_seconds": round(max(time.time() - started_ts, 0.0), 4),
                    },
                )

                print(
                    f"  sqlite inserted={result['inserted']} invalid={result['invalid']} "
                    f"last_line={result['last_line']} last_offset={result['last_offset_bytes']} "
                    f"pending_lines={pending_lines}"
                )
                slept = _ingest_cooldown_sleep(
                    base_sleep_seconds=max(float(args.ingest_file_sleep_seconds), 0.0),
                    host_load_soft_cap=max(float(args.ingest_host_load_soft_cap), 0.0),
                    host_load_sleep_seconds=max(float(args.ingest_host_load_sleep_seconds), 0.0),
                )
                if slept > 0.0:
                    cooldown_metrics["events"] += 1
                    cooldown_metrics["sleep_seconds"] += slept

            if args.mode in {"mysql", "both"}:
                progress = state["mysql"].get(rel, {"last_line": 0, "mtime": 0.0})
                start_line, start_offset, reset_reason = _derive_start_cursor(progress, st)
                _journal_event(
                    journal_paths,
                    {
                        "timestamp_utc": _now_utc(),
                        "event": "file_start",
                        "ingest_run_id": ingest_run_id,
                        "mode": "mysql",
                        "source_rel": rel,
                        "stream": stream,
                        "start_line": int(start_line),
                        "start_offset_bytes": int(start_offset),
                        "reset_reason": str(reset_reason),
                    },
                )

                lat_all, lat_stream = _ensure_latency_bucket(latency_metrics, "mysql", stream)
                started_ts = time.time()
                try:
                    result = _sync_file_to_mysql(
                        args.mysql_bin,
                        args.mysql_host,
                        args.mysql_port,
                        args.mysql_user,
                        args.mysql_password,
                        args.mysql_database,
                        args.mysql_table,
                        project_root,
                        fp,
                        start_line,
                        start_offset,
                        args.mysql_batch_size,
                        args.dry_run,
                        latency_all=lat_all,
                        latency_stream=lat_stream,
                        invalid_log_path=invalid_log_path,
                        invalid_sample_limit=max(args.invalid_sample_limit, 0),
                        run_id=run_id,
                        iter_id=iter_id,
                        max_lines_per_file=max(int(args.max_lines_per_file), 0),
                        max_bytes_per_file=max(int(args.max_bytes_per_file), 0),
                        oversize_payload_bytes=max(int(args.oversize_payload_bytes), 0),
                    )
                except FileNotFoundError:
                    _journal_event(
                        journal_paths,
                        {
                            "timestamp_utc": _now_utc(),
                            "event": "file_failed",
                            "ingest_run_id": ingest_run_id,
                            "mode": "mysql",
                            "source_rel": rel,
                            "stream": stream,
                            "error": "file_vanished_during_sync",
                        },
                    )
                    print(f"  mysql skipped vanished file during sync: {rel}")
                    continue
                except Exception as exc:
                    _journal_event(
                        journal_paths,
                        {
                            "timestamp_utc": _now_utc(),
                            "event": "file_failed",
                            "ingest_run_id": ingest_run_id,
                            "mode": "mysql",
                            "source_rel": rel,
                            "stream": stream,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise

                total_inserted["mysql"] += int(result["inserted"])
                total_invalid["mysql"] += int(result["invalid"])
                total_invalid_samples["mysql"] += int(result["invalid_samples_logged"])
                lag_metrics["mysql"]["oversize_payloads"] = int(
                    lag_metrics["mysql"].get("oversize_payloads", 0) or 0
                ) + int(result.get("oversize_payloads", 0) or 0)

                try:
                    post_st = fp.stat()
                except Exception:
                    post_st = st

                state["mysql"][rel] = {
                    "last_line": int(result["last_line"]),
                    "last_offset_bytes": int(result["last_offset_bytes"]),
                    "mtime": float(post_st.st_mtime),
                    "file_inode": int(post_st.st_ino),
                    "file_size_bytes": int(post_st.st_size),
                }

                pending_lines = max(int(total_lines) - int(result["last_line"]), 0)
                oldest_age = max(time.time() - mtime, 0.0) if pending_lines > 0 else 0.0
                lag_metrics["mysql"]["pending_lines"] += pending_lines
                lag_metrics["mysql"]["oldest_uningested_age_seconds"] = max(
                    float(lag_metrics["mysql"].get("oldest_uningested_age_seconds", 0.0) or 0.0),
                    float(oldest_age),
                )
                if pending_lines > 0:
                    lag_metrics["mysql"]["files_with_pending"] += 1
                    _record_top_pending(
                        lag_metrics["mysql"]["top_pending_files"],
                        source_rel=rel,
                        pending_lines=pending_lines,
                        oldest_age_seconds=oldest_age,
                        total_lines=total_lines,
                        last_line=int(result["last_line"]),
                        top_n=max(int(args.top_pending_files), 1),
                    )

                _journal_event(
                    journal_paths,
                    {
                        "timestamp_utc": _now_utc(),
                        "event": "file_complete",
                        "ingest_run_id": ingest_run_id,
                        "mode": "mysql",
                        "source_rel": rel,
                        "stream": stream,
                        "inserted": int(result["inserted"]),
                        "invalid": int(result["invalid"]),
                        "invalid_samples_logged": int(result["invalid_samples_logged"]),
                        "oversize_payloads": int(result.get("oversize_payloads", 0) or 0),
                        "last_line": int(result["last_line"]),
                        "last_offset_bytes": int(result["last_offset_bytes"]),
                        "pending_lines": int(pending_lines),
                        "duration_seconds": round(max(time.time() - started_ts, 0.0), 4),
                    },
                )

                print(
                    f"  mysql inserted={result['inserted']} invalid={result['invalid']} "
                    f"last_line={result['last_line']} last_offset={result['last_offset_bytes']} "
                    f"pending_lines={pending_lines}"
                )

        if args.mode in {"sqlite", "both"}:
            for fp in json_files:
                rel = str(fp.relative_to(project_root))
                try:
                    st = fp.stat()
                except FileNotFoundError:
                    print(f"Skipping vanished JSON file before sync: {rel}")
                    continue

                progress = state["sqlite_json_files"].get(rel, {})
                current_sig = (
                    int(getattr(st, "st_ino", 0)),
                    int(getattr(st, "st_size", 0)),
                    float(getattr(st, "st_mtime", 0.0)),
                )
                previous_sig = (
                    int(progress.get("file_inode", 0) or 0),
                    int(progress.get("file_size_bytes", 0) or 0),
                    float(progress.get("mtime", 0.0) or 0.0),
                )
                if current_sig == previous_sig:
                    json_file_metrics["sqlite"]["skipped_unchanged"] += 1
                    continue

                print(f"Syncing JSON file: {rel}")
                _journal_event(
                    journal_paths,
                    {
                        "timestamp_utc": _now_utc(),
                        "event": "json_file_start",
                        "ingest_run_id": ingest_run_id,
                        "mode": "sqlite",
                        "source_rel": rel,
                        "stream": _json_file_stream(rel),
                    },
                )

                try:
                    result = _sync_json_file_to_sqlite(
                        sqlite_conn,
                        args.sqlite_json_table,
                        project_root,
                        fp,
                        args.dry_run,
                        lock_retries=max(args.sqlite_lock_retries, 0),
                        lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                    )
                except Exception as exc:
                    json_file_metrics["sqlite"]["invalid"] += 1
                    _journal_event(
                        journal_paths,
                        {
                            "timestamp_utc": _now_utc(),
                            "event": "json_file_failed",
                            "ingest_run_id": ingest_run_id,
                            "mode": "sqlite",
                            "source_rel": rel,
                            "stream": _json_file_stream(rel),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    print(f"  sqlite json file failed: {rel} error={exc}")
                    continue

                if not args.dry_run and sqlite_conn is not None:
                    sqlite_conn.commit()

                post_st = fp.stat()
                state["sqlite_json_files"][rel] = {
                    "mtime": float(post_st.st_mtime),
                    "file_inode": int(post_st.st_ino),
                    "file_size_bytes": int(post_st.st_size),
                    "payload_sha1": str(result["payload_sha1"]),
                }
                json_file_metrics["sqlite"]["inserted"] += int(result["inserted"])
                json_file_metrics["sqlite"]["bytes"] += int(result["payload_size_bytes"])

                _journal_event(
                    journal_paths,
                    {
                        "timestamp_utc": _now_utc(),
                        "event": "json_file_complete",
                        "ingest_run_id": ingest_run_id,
                        "mode": "sqlite",
                        "source_rel": rel,
                        "stream": str(result["stream"]),
                        "inserted": int(result["inserted"]),
                        "payload_size_bytes": int(result["payload_size_bytes"]),
                    },
                )

                print(
                    f"  sqlite json inserted={result['inserted']} "
                    f"payload_size_bytes={result['payload_size_bytes']}"
                )

        if not args.dry_run:
            _save_state(state_path, state)

        sqlite_sink_enabled = args.mode in {"sqlite", "both"}
        mysql_sink_enabled = args.mode in {"mysql", "both"}
        health_payload = {
            "timestamp_utc": _now_utc(),
            "log_schema_version": _log_schema_version(),
            "run_id": run_id,
            "iter_id": iter_id,
            "ingest_run_id": ingest_run_id,
            "mode": args.mode,
            "sinks": {
                "sqlite": {
                    "enabled": bool(sqlite_sink_enabled),
                    "status": "active" if sqlite_sink_enabled else "disabled_by_link_mode",
                },
                "mysql": {
                    "enabled": bool(mysql_sink_enabled),
                    "status": "active" if mysql_sink_enabled else "disabled_by_link_mode",
                },
            },
            "project_root": str(project_root),
            "state_file": str(state_path),
            "health_file": str(health_file_path),
            "checkpoint_mode": "line_offset_inode_v2",
            "files_discovered": int(len(files)),
            "json_files_discovered": int(len(json_files)),
            "invalid_log_file": str(invalid_log_path),
            "journal_files": [str(p) for p in journal_paths],
            "filters": {
                "include_streams": include_streams,
                "exclude_streams": exclude_streams,
                "path_contains": path_contains,
                "path_not_contains": path_not_contains,
                "max_lines_per_file": max(int(args.max_lines_per_file), 0),
                "max_bytes_per_file": max(int(args.max_bytes_per_file), 0),
                "oversize_payload_bytes": max(int(args.oversize_payload_bytes), 0),
                "sqlite_batch_max_bytes": max(int(args.sqlite_batch_max_bytes), 0),
            },
            "ingest_cooling_policy": {
                "flush_sleep_seconds": max(float(args.ingest_flush_sleep_seconds), 0.0),
                "file_sleep_seconds": max(float(args.ingest_file_sleep_seconds), 0.0),
                "host_load_soft_cap": max(float(args.ingest_host_load_soft_cap), 0.0),
                "host_load_sleep_seconds": max(float(args.ingest_host_load_sleep_seconds), 0.0),
                "cooldown_events": int(cooldown_metrics["events"]),
                "cooldown_sleep_seconds": round(float(cooldown_metrics["sleep_seconds"]), 3),
                "policy": "yield_jsonl_sql_catchup_when_host_load_is_hot",
            },
            "sqlite": {
                "enabled": bool(sqlite_sink_enabled),
                "status": "active" if sqlite_sink_enabled else "disabled_by_link_mode",
                "inserted": int(total_inserted["sqlite"]),
                "invalid": int(total_invalid["sqlite"]),
                "invalid_samples_logged": int(total_invalid_samples["sqlite"]),
                "oversize_payloads": int(lag_metrics["sqlite"].get("oversize_payloads", 0) or 0),
                "ops_write_failures": int(lag_metrics["sqlite"].get("ops_write_failures", 0) or 0),
                "pending_lines": int(lag_metrics["sqlite"]["pending_lines"]),
                "oldest_uningested_age_seconds": float(lag_metrics["sqlite"]["oldest_uningested_age_seconds"]),
                "files_with_pending": int(lag_metrics["sqlite"]["files_with_pending"]),
                "top_pending_files": list(lag_metrics["sqlite"]["top_pending_files"]),
            },
            "sqlite_json_files": {
                "inserted": int(json_file_metrics["sqlite"]["inserted"]),
                "invalid": int(json_file_metrics["sqlite"]["invalid"]),
                "skipped_unchanged": int(json_file_metrics["sqlite"]["skipped_unchanged"]),
                "bytes": int(json_file_metrics["sqlite"]["bytes"]),
            },
            "mysql": {
                "enabled": bool(mysql_sink_enabled),
                "status": "active" if mysql_sink_enabled else "disabled_by_link_mode",
                "inserted": int(total_inserted["mysql"]),
                "invalid": int(total_invalid["mysql"]),
                "invalid_samples_logged": int(total_invalid_samples["mysql"]),
                "oversize_payloads": int(lag_metrics["mysql"].get("oversize_payloads", 0) or 0),
                "pending_lines": int(lag_metrics["mysql"]["pending_lines"]),
                "oldest_uningested_age_seconds": float(lag_metrics["mysql"]["oldest_uningested_age_seconds"]),
                "files_with_pending": int(lag_metrics["mysql"]["files_with_pending"]),
                "top_pending_files": list(lag_metrics["mysql"]["top_pending_files"]),
            },
            "latency_slo": {
                "sqlite": _latency_payload(
                    latency_metrics["sqlite"]["all"],
                    latency_metrics["sqlite"]["by_stream"],
                ),
                "mysql": _latency_payload(
                    latency_metrics["mysql"]["all"],
                    latency_metrics["mysql"]["by_stream"],
                ),
            },
        }

        health_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(health_file_path, "w", encoding="utf-8") as f:
            json.dump(health_payload, f, ensure_ascii=True, indent=2)

        print("Done.")
        if args.mode in {"sqlite", "both"}:
            sqlite_lat = health_payload["latency_slo"]["sqlite"]["all"]
            print(
                f"SQLite total inserted={total_inserted['sqlite']} invalid={total_invalid['sqlite']} "
                f"pending={lag_metrics['sqlite']['pending_lines']} "
                f"oldest_pending_age_s={lag_metrics['sqlite']['oldest_uningested_age_seconds']:.1f} "
                f"p95_latency_s={float(sqlite_lat.get('p95_seconds', 0.0) or 0.0):.1f}"
            )
            print(
                f"SQLite JSON files inserted={json_file_metrics['sqlite']['inserted']} "
                f"invalid={json_file_metrics['sqlite']['invalid']} "
                f"skipped_unchanged={json_file_metrics['sqlite']['skipped_unchanged']} "
                f"bytes={json_file_metrics['sqlite']['bytes']}"
            )
        if args.mode in {"mysql", "both"}:
            mysql_lat = health_payload["latency_slo"]["mysql"]["all"]
            print(
                f"MySQL total inserted={total_inserted['mysql']} invalid={total_invalid['mysql']} "
                f"pending={lag_metrics['mysql']['pending_lines']} "
                f"oldest_pending_age_s={lag_metrics['mysql']['oldest_uningested_age_seconds']:.1f} "
                f"p95_latency_s={float(mysql_lat.get('p95_seconds', 0.0) or 0.0):.1f}"
            )
        print(f"State file: {state_path}")
        print(f"Health summary: {health_file_path}")
        return 0
    finally:
        if sqlite_conn is not None:
            sqlite_conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
