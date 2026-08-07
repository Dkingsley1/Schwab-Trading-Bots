#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.profitability_statistics import clustered_post_cost_statistics
from scripts.ops.long_runtime_common import load_json, ordered_unique, parse_iso_utc, write_payload


DEFAULT_DB = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "canary_rollout_latest.json"
DEFAULT_CANDIDATE_STATE = PROJECT_ROOT / "governance" / "runtime" / "production_candidate_state.json"
DEFAULT_SCAN_STATE = Path("governance/runtime/canary_rollout_scan_state.json")
DEFAULT_EVIDENCE_CACHE = Path("governance/evidence/canary_rollout_observations.jsonl")
SCHEMA_VERSION = 3
PROFITABILITY_SCOPE_IDS = ("strategy", "execution", "risk", "data", "promotion", "dependencies")
PROFILE_ROWS_SQL = """
    SELECT
      source_rel,
      payload_sha1,
      COALESCE(
        json_extract(payload_json, '$.profile'),
        json_extract(payload_json, '$.shadow_profile'),
        'unknown'
      ) AS profile,
      COALESCE(json_extract(payload_json, '$.bot_id'), 'unknown') AS bot_id,
      COALESCE(json_extract(payload_json, '$.snapshot_id'), '') AS snapshot_id,
      COALESCE(json_extract(payload_json, '$.symbol'), 'UNKNOWN') AS symbol,
      COALESCE(json_extract(payload_json, '$.timestamp_utc'), '') AS timestamp_utc,
      CAST(COALESCE(json_extract(payload_json, '$.pnl_proxy'), 0.0) AS REAL) AS pnl_proxy,
      COALESCE(json_extract(payload_json, '$.action'), 'HOLD') AS action,
      COALESCE(json_extract(payload_json, '$.run_id'), '') AS run_id,
      COALESCE(json_extract(payload_json, '$.iter_id'), '') AS iter_id
    FROM jsonl_records
    WHERE source_rel GLOB ?
      AND COALESCE(
        json_extract(payload_json, '$.profile'),
        json_extract(payload_json, '$.shadow_profile'),
        'unknown'
      ) IN ({profile_placeholders})
"""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _csv(raw: str) -> list[str]:
    return ordered_unique([part.strip().lower() for part in str(raw or "").split(",")])


def _candidate_binding(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    windows = payload.get("scope_windows_started_utc") if isinstance(payload.get("scope_windows_started_utc"), dict) else {}
    parsed_windows = {
        scope_id: parsed
        for scope_id in PROFITABILITY_SCOPE_IDS
        if (parsed := parse_iso_utc(windows.get(scope_id))) is not None
    }
    cutoff = max(parsed_windows.values()) if parsed_windows else None
    cutoff_scope_ids = sorted(
        scope_id for scope_id, started in parsed_windows.items() if cutoff is not None and started == cutoff
    )
    return {
        "candidate_id": str(payload.get("candidate_id") or "").strip(),
        "generation": int(_safe_float(payload.get("generation"), 0.0)),
        "accepted_git_head": str(payload.get("accepted_git_head") or "").strip(),
        "promotion_window_started_utc": (
            parsed_windows["promotion"].isoformat() if "promotion" in parsed_windows else ""
        ),
        "profitability_window_started_utc": cutoff.isoformat() if cutoff is not None else "",
        "profitability_scope_ids": list(PROFITABILITY_SCOPE_IDS),
        "cutoff_scope_ids": cutoff_scope_ids,
        "scope_windows_started_utc": {
            scope_id: started.isoformat() for scope_id, started in sorted(parsed_windows.items())
        },
        "cutoff": cutoff,
        "bound": bool(str(payload.get("candidate_id") or "").strip() and cutoff is not None),
    }


def _iter_partition_days(start: datetime, end: datetime, *, padding_days: int = 1) -> Iterable[str]:
    # Runtime JSONL partitions use the host trading date while evidence windows
    # are UTC. Scan the adjacent labels and trust each row timestamp for truth.
    padding = max(int(padding_days), 0)
    current = start.date() - timedelta(days=padding)
    final = end.date() + timedelta(days=padding)
    while current <= final:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    stable = (
        str(row.get("profile") or ""),
        str(row.get("bot_id") or ""),
        str(row.get("snapshot_id") or ""),
        str(row.get("symbol") or ""),
        str(row.get("timestamp_utc") or ""),
    )
    if any(stable[2:]):
        return stable
    return (str(row.get("source_rel") or ""), str(row.get("payload_sha1") or ""))


def _load_db_rows(
    db_path: Path,
    *,
    start: datetime,
    end: datetime,
    profiles: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    placeholders = ",".join("?" for _ in profiles)
    sql = PROFILE_ROWS_SQL.format(profile_placeholders=placeholders)
    rows: list[dict[str, Any]] = []
    scanned = 0
    invalid_timestamp = 0
    before_cutoff = 0
    after_end = 0
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    partition_days = list(_iter_partition_days(start, end))
    rows_by_profile: dict[str, int] = {}
    conn = sqlite3.connect(str(db_path), timeout=15.0)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        for day in partition_days:
            source_glob = f"governance/*/shadow_pnl_attribution_{day}.jsonl"
            for raw in conn.execute(sql, (source_glob, *profiles)):
                scanned += 1
                timestamp = parse_iso_utc(raw[6])
                if timestamp is None:
                    invalid_timestamp += 1
                    continue
                if timestamp < start:
                    before_cutoff += 1
                    continue
                if timestamp > end:
                    after_end += 1
                    continue
                row = {
                    "source_rel": str(raw[0] or ""),
                    "payload_sha1": str(raw[1] or ""),
                    "profile": str(raw[2] or "unknown").strip().lower(),
                    "bot_id": str(raw[3] or "unknown").strip().lower(),
                    "snapshot_id": str(raw[4] or "").strip(),
                    "symbol": str(raw[5] or "UNKNOWN").strip().upper(),
                    "timestamp_utc": timestamp.isoformat(),
                    "pnl_proxy": _safe_float(raw[7]),
                    "action": str(raw[8] or "HOLD").strip().upper(),
                    "run_id": str(raw[9] or "").strip(),
                    "iter_id": str(raw[10] or "").strip(),
                }
                key = _row_key(row)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                rows.append(row)
                profile = str(row.get("profile") or "unknown")
                rows_by_profile[profile] = rows_by_profile.get(profile, 0) + 1
    finally:
        conn.close()
    return rows, {
        "source": "sqlite_link_fallback",
        "rows_scanned": scanned,
        "rows_retained": len(rows),
        "duplicates_removed": duplicates,
        "invalid_timestamp_rows": invalid_timestamp,
        "rows_before_candidate_cutoff": before_cutoff,
        "rows_after_end": after_end,
        "partition_day_labels": partition_days,
        "rows_retained_by_profile": rows_by_profile,
    }


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    serialized = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, serialized)


def _load_cached_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except Exception:
        return rows
    with handle:
        for raw in handle:
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _compact_source_row(payload: dict[str, Any], *, source_rel: str) -> dict[str, Any] | None:
    profile = str(payload.get("profile") or payload.get("shadow_profile") or "unknown").strip().lower()
    timestamp = parse_iso_utc(payload.get("timestamp_utc"))
    if timestamp is None:
        return None
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source_rel": source_rel,
        "payload_sha1": digest,
        "profile": profile,
        "bot_id": str(payload.get("bot_id") or "unknown").strip().lower(),
        "snapshot_id": str(payload.get("snapshot_id") or "").strip(),
        "symbol": str(payload.get("symbol") or "UNKNOWN").strip().upper(),
        "timestamp_utc": timestamp.isoformat(),
        "pnl_proxy": _safe_float(payload.get("pnl_proxy")),
        "action": str(payload.get("action") or "HOLD").strip().upper(),
        "run_id": str(payload.get("run_id") or "").strip(),
        "iter_id": str(payload.get("iter_id") or "").strip(),
    }


def _filesystem_rows(
    project_root: Path,
    *,
    candidate_id: str,
    candidate_generation: int,
    candidate_cutoff: datetime | None,
    start: datetime,
    end: datetime,
    profiles: list[str],
    state_path: Path,
    evidence_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_state = load_json(state_path)
    normalized_profiles = sorted(ordered_unique(profiles))
    prior_profiles = sorted(
        ordered_unique(prior_state.get("profiles") if isinstance(prior_state.get("profiles"), list) else [])
    )
    candidate_cutoff_utc = candidate_cutoff.isoformat() if candidate_cutoff is not None else ""
    candidate_metadata_changed = bool(
        str(prior_state.get("candidate_id") or "") != candidate_id
        or int(_safe_float(prior_state.get("candidate_generation"), 0.0)) != int(candidate_generation)
    )
    evidence_window_changed = bool(
        str(prior_state.get("candidate_cutoff_utc") or "") != candidate_cutoff_utc
        or prior_profiles != normalized_profiles
    )
    binding_changed = bool(candidate_metadata_changed or evidence_window_changed)
    cached_rows = [] if evidence_window_changed else _load_cached_rows(evidence_path)
    cached_rows_loaded = len(cached_rows)
    cached_rows = [
        row
        for row in cached_rows
        if (timestamp := parse_iso_utc(row.get("timestamp_utc"))) is not None and start <= timestamp <= end
    ]
    cached_rows_pruned = cached_rows_loaded - len(cached_rows)
    seen = {_row_key(row) for row in cached_rows}
    prior_files = (
        prior_state.get("files")
        if isinstance(prior_state.get("files"), dict) and not evidence_window_changed
        else {}
    )
    next_files: dict[str, Any] = {}
    new_rows: list[dict[str, Any]] = []
    files_seen = 0
    files_advanced = 0
    bytes_read = 0
    invalid_json_rows = 0
    invalid_timestamp_rows = 0
    rows_before_cutoff = 0
    duplicates_removed = 0
    files_seen_by_profile: dict[str, int] = {}
    files_advanced_by_profile: dict[str, int] = {}
    governance_root = project_root / "governance"
    profile_set = set(profiles)
    partition_days = list(_iter_partition_days(start, end))
    for profile in profiles:
        for day in partition_days:
            pattern = f"shadow_{profile}_*/shadow_pnl_attribution_{day}.jsonl"
            for path in sorted(governance_root.glob(pattern)):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                files_seen += 1
                files_seen_by_profile[profile] = files_seen_by_profile.get(profile, 0) + 1
                relative = str(path.relative_to(project_root))
                previous = prior_files.get(relative) if isinstance(prior_files.get(relative), dict) else {}
                same_file = bool(
                    _safe_int(previous.get("device"), -1) == int(stat.st_dev)
                    and _safe_int(previous.get("inode"), -1) == int(stat.st_ino)
                    and _safe_int(previous.get("offset"), 0) <= int(stat.st_size)
                )
                offset = _safe_int(previous.get("offset"), 0) if same_file else 0
                last_complete_offset = offset
                if int(stat.st_size) > offset:
                    files_advanced += 1
                    files_advanced_by_profile[profile] = files_advanced_by_profile.get(profile, 0) + 1
                try:
                    handle = path.open("rb")
                except OSError:
                    continue
                with handle:
                    handle.seek(offset)
                    while True:
                        line_start = handle.tell()
                        raw = handle.readline()
                        if not raw:
                            last_complete_offset = handle.tell()
                            break
                        bytes_read += len(raw)
                        if not raw.endswith(b"\n"):
                            last_complete_offset = line_start
                            break
                        last_complete_offset = handle.tell()
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            invalid_json_rows += 1
                            continue
                        if not isinstance(payload, dict):
                            invalid_json_rows += 1
                            continue
                        row = _compact_source_row(payload, source_rel=relative)
                        if row is None:
                            invalid_timestamp_rows += 1
                            continue
                        timestamp = parse_iso_utc(row.get("timestamp_utc"))
                        if timestamp is None:
                            invalid_timestamp_rows += 1
                            continue
                        if timestamp < start:
                            rows_before_cutoff += 1
                            continue
                        if timestamp > end or str(row.get("profile")) not in profile_set:
                            continue
                        key = _row_key(row)
                        if key in seen:
                            duplicates_removed += 1
                            continue
                        seen.add(key)
                        new_rows.append(row)
                next_files[relative] = {
                    "device": int(stat.st_dev),
                    "inode": int(stat.st_ino),
                    "offset": int(last_complete_offset),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
    rows = sorted(
        cached_rows + new_rows,
        key=lambda row: (str(row.get("timestamp_utc") or ""), str(row.get("profile") or ""), str(row.get("bot_id") or ""), str(row.get("symbol") or "")),
    )
    evidence_cache_changed = bool(
        evidence_window_changed or cached_rows_pruned or new_rows or next_files != prior_files
    )
    if evidence_cache_changed:
        _atomic_write_jsonl(evidence_path, rows)
    if binding_changed or evidence_cache_changed:
        write_payload(
            state_path,
            {
                "schema_version": 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_id": candidate_id,
                "candidate_generation": int(candidate_generation),
                "candidate_cutoff_utc": candidate_cutoff_utc,
                "effective_scan_started_utc": start.isoformat(),
                "profiles": normalized_profiles,
                "files": next_files,
                "cached_row_count": len(rows),
            },
        )
    return rows, {
        "source": "incremental_jsonl_evidence_cache",
        "binding_changed": binding_changed,
        "candidate_metadata_changed": candidate_metadata_changed,
        "evidence_window_changed": evidence_window_changed,
        "valid_cache_reused_across_candidate_metadata_change": bool(
            candidate_metadata_changed and not evidence_window_changed
        ),
        "files_seen": files_seen,
        "files_advanced": files_advanced,
        "bytes_read": bytes_read,
        "cached_rows_loaded": cached_rows_loaded,
        "cached_rows_pruned": cached_rows_pruned,
        "cached_rows_before": len(cached_rows),
        "new_rows": len(new_rows),
        "rows_retained": len(rows),
        "duplicates_removed": duplicates_removed,
        "invalid_json_rows": invalid_json_rows,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "rows_before_candidate_cutoff": rows_before_cutoff,
        "partition_day_labels": partition_days,
        "files_seen_by_profile": files_seen_by_profile,
        "files_advanced_by_profile": files_advanced_by_profile,
        "state_path": str(state_path),
        "evidence_path": str(evidence_path),
    }


def _statistics_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "strategy": str(row.get("profile") or "unknown"),
            "post_cost_pnl_delta": _safe_float(row.get("pnl_proxy")),
            "post_cost_return_bps": _safe_float(row.get("pnl_proxy")) * 10000.0,
        }
        for row in rows
    ]


def _cohort_statistics(
    rows: list[dict[str, Any]],
    *,
    minimum_samples: int,
    minimum_days: int,
    minimum_symbols: int,
    minimum_effective_samples: float,
) -> dict[str, Any]:
    return clustered_post_cost_statistics(
        _statistics_rows(rows),
        minimum_samples=minimum_samples,
        minimum_days=minimum_days,
        minimum_symbols=minimum_symbols,
        minimum_effective_samples=minimum_effective_samples,
        hypothesis_count=2,
        bootstrap_iterations=600,
    )


def _edge_statistics(canary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    canary_mean = _safe_float(canary.get("mean_post_cost_pnl_delta"))
    baseline_mean = _safe_float(baseline.get("mean_post_cost_pnl_delta"))
    canary_se = _safe_float(canary.get("conservative_standard_error_post_cost_pnl_delta"))
    baseline_se = _safe_float(baseline.get("conservative_standard_error_post_cost_pnl_delta"))
    edge = canary_mean - baseline_mean
    edge_se = math.sqrt(canary_se * canary_se + baseline_se * baseline_se)
    return {
        "edge_delta": round(edge, 10),
        "conservative_standard_error": round(edge_se, 10),
        "lower_confidence_bound_95": round(edge - 1.96 * edge_se, 10),
        "method": "difference_of_candidate_bound_clustered_cohort_means",
    }


def _cohort_source_coverage(
    *,
    profiles: list[str],
    rows: list[dict[str, Any]],
    filesystem_scan: dict[str, Any],
    db_scan: dict[str, Any],
) -> dict[str, Any]:
    files_by_profile = filesystem_scan.get("files_seen_by_profile")
    files_by_profile = files_by_profile if isinstance(files_by_profile, dict) else {}
    db_rows_by_profile = db_scan.get("rows_retained_by_profile")
    db_rows_by_profile = db_rows_by_profile if isinstance(db_rows_by_profile, dict) else {}
    candidate_rows_by_profile: dict[str, int] = {}
    for row in rows:
        profile = str(row.get("profile") or "").strip().lower()
        if profile:
            candidate_rows_by_profile[profile] = candidate_rows_by_profile.get(profile, 0) + 1
    source_profiles = sorted(
        profile
        for profile in profiles
        if _safe_int(files_by_profile.get(profile), 0) > 0 or _safe_int(db_rows_by_profile.get(profile), 0) > 0
    )
    row_profiles = sorted(profile for profile in profiles if candidate_rows_by_profile.get(profile, 0) > 0)
    missing_source_profiles = sorted(set(profiles) - set(source_profiles))
    return {
        "required_profiles": profiles,
        "source_profiles": source_profiles,
        "candidate_row_profiles": row_profiles,
        "missing_source_profiles": missing_source_profiles,
        "source_ready": not missing_source_profiles,
        "files_seen_by_profile": {profile: _safe_int(files_by_profile.get(profile), 0) for profile in profiles},
        "candidate_rows_by_profile": {profile: candidate_rows_by_profile.get(profile, 0) for profile in profiles},
    }


def build_payload(
    *,
    db_path: Path,
    candidate_state_path: Path,
    end: datetime,
    lookback_days: int,
    canary_profiles: list[str],
    baseline_profiles: list[str],
    minimum_samples: int,
    minimum_days: int,
    minimum_symbols: int,
    minimum_effective_samples: float,
    minimum_edge_delta: float,
    scan_state_path: Path | None = None,
    evidence_cache_path: Path | None = None,
) -> dict[str, Any]:
    binding = _candidate_binding(candidate_state_path)
    lookback_start = end - timedelta(days=max(int(lookback_days), 1))
    candidate_cutoff = binding.get("cutoff") if isinstance(binding.get("cutoff"), datetime) else None
    start = max([value for value in (lookback_start, candidate_cutoff) if value is not None])
    profiles = ordered_unique(canary_profiles + baseline_profiles)
    project_root = candidate_state_path.resolve().parents[2]
    effective_scan_state = scan_state_path or project_root / DEFAULT_SCAN_STATE
    effective_evidence_cache = evidence_cache_path or project_root / DEFAULT_EVIDENCE_CACHE
    filesystem_rows, filesystem_scan = _filesystem_rows(
        project_root,
        candidate_id=str(binding.get("candidate_id") or ""),
        candidate_generation=int(_safe_float(binding.get("generation"), 0.0)),
        candidate_cutoff=candidate_cutoff,
        start=start,
        end=end,
        profiles=profiles,
        state_path=effective_scan_state,
        evidence_path=effective_evidence_cache,
    )
    files_seen_by_profile = (
        filesystem_scan.get("files_seen_by_profile")
        if isinstance(filesystem_scan.get("files_seen_by_profile"), dict)
        else {}
    )
    filesystem_source_profiles = {
        profile for profile in profiles if _safe_int(files_seen_by_profile.get(profile), 0) > 0
    }
    missing_profiles = [profile for profile in profiles if profile not in filesystem_source_profiles]
    if missing_profiles:
        db_rows, db_scan = _load_db_rows(db_path, start=start, end=end, profiles=missing_profiles)
    else:
        db_rows, db_scan = [], {
            "source": "sqlite_link_fallback",
            "skipped": True,
            "reason": "all_profiles_have_authoritative_jsonl_sources",
            "rows_retained_by_profile": {},
        }
    combined: list[dict[str, Any]] = []
    combined_seen: set[tuple[str, ...]] = set()
    cross_source_duplicates = 0
    for row in filesystem_rows + db_rows:
        key = _row_key(row)
        if key in combined_seen:
            cross_source_duplicates += 1
            continue
        combined_seen.add(key)
        combined.append(row)
    rows = combined
    scan = {
        "primary": filesystem_scan,
        "fallback": db_scan,
        "rows_retained": len(rows),
        "duplicates_removed": int(filesystem_scan.get("duplicates_removed", 0))
        + int(db_scan.get("duplicates_removed", 0))
        + cross_source_duplicates,
        "cross_source_duplicates_removed": cross_source_duplicates,
        "filesystem_source_profiles": sorted(filesystem_source_profiles),
        "policy": "incremental source JSONL is authoritative; the linked SQLite database fills only missing profiles",
    }
    canary_rows = [row for row in rows if str(row.get("profile")) in set(canary_profiles)]
    baseline_rows = [row for row in rows if str(row.get("profile")) in set(baseline_profiles)]
    source_coverage = {
        "canary": _cohort_source_coverage(
            profiles=canary_profiles,
            rows=canary_rows,
            filesystem_scan=filesystem_scan,
            db_scan=db_scan,
        ),
        "baseline": _cohort_source_coverage(
            profiles=baseline_profiles,
            rows=baseline_rows,
            filesystem_scan=filesystem_scan,
            db_scan=db_scan,
        ),
    }
    canary_stats = _cohort_statistics(
        canary_rows,
        minimum_samples=minimum_samples,
        minimum_days=minimum_days,
        minimum_symbols=minimum_symbols,
        minimum_effective_samples=minimum_effective_samples,
    )
    baseline_stats = _cohort_statistics(
        baseline_rows,
        minimum_samples=minimum_samples,
        minimum_days=minimum_days,
        minimum_symbols=minimum_symbols,
        minimum_effective_samples=minimum_effective_samples,
    )
    edge = _edge_statistics(canary_stats, baseline_stats)
    cohort_evidence_ready = bool(
        canary_stats.get("promotion_evidence_sufficient", False)
        and baseline_stats.get("promotion_evidence_sufficient", False)
    )
    eligible = bool(binding.get("bound", False) and cohort_evidence_ready)
    promote = bool(
        eligible
        and _safe_float(edge.get("edge_delta")) >= float(minimum_edge_delta)
        and _safe_float(edge.get("lower_confidence_bound_95")) > float(minimum_edge_delta)
    )
    blockers: list[str] = []
    if not binding.get("bound", False):
        blockers.append("production_candidate_binding_missing")
    blockers.extend(f"canary_{item}" for item in canary_stats.get("blockers") or [])
    blockers.extend(f"baseline_{item}" for item in baseline_stats.get("blockers") or [])
    if cohort_evidence_ready and _safe_float(edge.get("lower_confidence_bound_95")) <= float(minimum_edge_delta):
        blockers.append("canary_edge_lower_confidence_bound_not_positive")
    canary_average = _safe_float(canary_stats.get("mean_post_cost_pnl_delta"))
    baseline_average = _safe_float(baseline_stats.get("mean_post_cost_pnl_delta"))
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "ready" if promote else "collecting" if binding.get("bound", False) else "blocked",
        "ok": bool(binding.get("bound", False)),
        "day": end.strftime("%Y%m%d"),
        "candidate_binding": {key: value for key, value in binding.items() if key != "cutoff"},
        "evidence_window": {
            "started_utc": start.isoformat(),
            "ended_utc": end.isoformat(),
            "lookback_days": max(int(lookback_days), 1),
            "candidate_cutoff_enforced": candidate_cutoff is not None,
        },
        "canary_profiles": canary_profiles,
        "baseline_profiles": baseline_profiles,
        "cohort_source_coverage": source_coverage,
        "canary_samples": len(canary_rows),
        "baseline_samples": len(baseline_rows),
        "canary_avg_pnl_proxy": round(canary_average, 10),
        "baseline_avg_pnl_proxy": round(baseline_average, 10),
        "edge_delta": edge["edge_delta"],
        "edge_statistics": edge,
        "canary_statistics": canary_stats,
        "baseline_statistics": baseline_stats,
        "eligible": eligible,
        "promote_canary": promote,
        "applied_weight": 0.01 if promote else 0.0025,
        "blockers": ordered_unique(blockers),
        "thresholds": {
            "minimum_samples_per_cohort": max(int(minimum_samples), 1),
            "minimum_independent_days": max(int(minimum_days), 1),
            "minimum_symbols": max(int(minimum_symbols), 1),
            "minimum_effective_samples": max(float(minimum_effective_samples), 1.0),
            "minimum_edge_delta": float(minimum_edge_delta),
        },
        "scan": scan,
        "control_contract": {
            "profile_field_prefers_schema_v2_profile": True,
            "legacy_shadow_profile_fallback": True,
            "candidate_window_enforced": True,
            "profitability_scope_windows_enforced": list(PROFITABILITY_SCOPE_IDS),
            "newest_profitability_scope_window_wins": True,
            "metadata_only_candidate_changes_preserve_valid_scan_state": True,
            "duplicate_observations_excluded": True,
            "raw_row_count_is_not_independent_evidence": True,
            "positive_clustered_edge_lcb_required": True,
            "utc_and_host_date_partition_boundaries_scanned": True,
            "source_coverage_reported_per_cohort": True,
            "live_execution_authority": False,
        },
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Candidate-bound canary rollout evidence guard.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--candidate-state", type=Path, default=DEFAULT_CANDIDATE_STATE)
    parser.add_argument("--scan-state", type=Path, default=DEFAULT_SCAN_STATE)
    parser.add_argument("--evidence-cache", type=Path, default=DEFAULT_EVIDENCE_CACHE)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--canary-profiles", default="intraday_aggressive,swing_aggressive")
    parser.add_argument("--baseline-profiles", default="conservative,aggressive")
    parser.add_argument("--min-samples", type=int, default=400)
    parser.add_argument("--min-independent-days", type=int, default=3)
    parser.add_argument("--min-symbols", type=int, default=5)
    parser.add_argument("--min-effective-samples", type=float, default=50.0)
    parser.add_argument("--min-edge-delta", type=float, default=0.0)
    parser.add_argument("--apply-env", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.db.exists():
        parser.error(f"SQLite DB not found: {args.db}")
    try:
        end = datetime.strptime(str(args.day), "%Y%m%d").replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(microseconds=1)
    except ValueError:
        parser.error("--day must use YYYYMMDD")
    canary_profiles = _csv(args.canary_profiles)
    baseline_profiles = _csv(args.baseline_profiles)
    if not canary_profiles or not baseline_profiles:
        parser.error("both canary and baseline profile cohorts are required")
    overlap = sorted(set(canary_profiles).intersection(baseline_profiles))
    if overlap:
        parser.error(f"canary and baseline cohorts must be disjoint: {','.join(overlap)}")

    payload = build_payload(
        db_path=args.db.expanduser(),
        candidate_state_path=args.candidate_state.expanduser(),
        end=end,
        lookback_days=int(args.lookback_days),
        canary_profiles=canary_profiles,
        baseline_profiles=baseline_profiles,
        minimum_samples=int(args.min_samples),
        minimum_days=int(args.min_independent_days),
        minimum_symbols=int(args.min_symbols),
        minimum_effective_samples=float(args.min_effective_samples),
        minimum_edge_delta=float(args.min_edge_delta),
        scan_state_path=(args.scan_state if args.scan_state.is_absolute() else PROJECT_ROOT / args.scan_state),
        evidence_cache_path=(args.evidence_cache if args.evidence_cache.is_absolute() else PROJECT_ROOT / args.evidence_cache),
    )
    write_payload(args.out_file.expanduser(), payload)
    if args.apply_env:
        env_file = PROJECT_ROOT / "governance" / "health" / "canary_rollout.env"
        _atomic_write_text(env_file, f"CANARY_MAX_WEIGHT={float(payload['applied_weight']):.4f}\n")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "canary_rollout "
            f"status={payload['overall_status']} eligible={int(payload['eligible'])} "
            f"promote={int(payload['promote_canary'])} edge_lcb={payload['edge_statistics']['lower_confidence_bound_95']:.8f} "
            f"samples_canary={payload['canary_samples']} samples_baseline={payload['baseline_samples']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
