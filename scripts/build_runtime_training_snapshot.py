#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import gc
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import runtime_training_common as rtc


DEFAULT_ROWS_PATH = PROJECT_ROOT / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
DEFAULT_HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_training_snapshot_latest.json"
_FILE_HASH_CHUNK_BYTES = 1024 * 1024


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
    payload = dict(summary)
    payload["reused"] = True
    payload["reuse_reason"] = "fresh_compatible_snapshot"
    payload["age_minutes"] = round(float(age_minutes), 4)
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


def _iter_json_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        try:
            if path.suffix == ".gz":
                handle_cm = gzip.open(path, "rt", encoding="utf-8")
            else:
                handle_cm = path.open("r", encoding="utf-8")
            with handle_cm as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        yield row
        except Exception:
            continue


def _normalize_runtime_observation(
    row: dict[str, Any],
    *,
    since_utc: datetime,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
) -> tuple[tuple[str, str], dict[str, Any]] | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if str(metadata.get("layer") or "").strip().lower() != "grand_master":
        return None
    strategy = str(row.get("strategy") or "").strip().lower()
    if strategy not in rtc._ROOT_STRATEGY_PRIORITY:
        return None
    ts = _parse_ts(row.get("timestamp_utc"))
    if ts is None or ts < since_utc:
        return None
    mode = str(row.get("mode") or "").strip().lower()
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

    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    price = rtc._safe_float(features.get("last_price"), 0.0)
    if price <= 0.0:
        return None

    snapshot_id = str(metadata.get("snapshot_id") or row.get("snapshot_id") or row.get("parent_decision_id") or "").strip()
    if not snapshot_id:
        snapshot_id = f"{symbol}:{ts.isoformat()}"

    obs = {
        "timestamp_utc": ts.isoformat(),
        "strategy": strategy,
        "strategy_priority": rtc._ROOT_STRATEGY_PRIORITY[strategy],
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
) -> int:
    best_by_snapshot: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in _iter_json_rows(candidate_paths):
        normalized = _normalize_runtime_observation(
            row,
            since_utc=since_utc,
            mode_allowlist=mode_allowlist,
            symbol_allowlist=symbol_allowlist,
        )
        if normalized is None:
            continue
        (mode, symbol), obs = normalized
        key = (mode, symbol, str(obs.get("snapshot_id") or ""))
        prev = best_by_snapshot.get(key)
        if prev is None or int(obs.get("strategy_priority", 99)) < int(prev.get("strategy_priority", 99)):
            best_by_snapshot[key] = obs

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
    return int(merged_row_count)


def _incremental_snapshot_sequences(
    summary: dict[str, Any],
    *,
    project_root: Path,
    health_path: Path,
    lookback_days: int,
    mode_allowlist: list[str],
    symbol_allowlist: list[str],
    prefer_sqlite: bool,
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
    since_summary_utc = _parse_ts(summary.get("timestamp_utc"))
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
    merged_row_count = _merge_candidate_rows_into_sequences(
        base_sequences,
        candidate_paths=candidate_paths,
        project_root=project_root,
        since_utc=since_utc,
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
    )

    return base_sequences, {
        "build_mode": "incremental_refresh",
        "incremental_base_timestamp_utc": since_summary_utc.isoformat(),
        "incremental_source_count": len(candidate_paths),
        "incremental_source_paths": [str(path) for path in candidate_paths[:20]],
        "incremental_row_count": int(merged_row_count),
    }


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

    merged_row_count = _merge_candidate_rows_into_sequences(
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
    }


def _coverage_summary(sequences: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    mode_row_counts: dict[str, int] = {}
    mode_sequence_counts: dict[str, int] = {}
    symbol_row_counts: dict[str, int] = {}
    sequence_rows: list[dict[str, Any]] = []
    for (mode, symbol), rows in sorted(sequences.items()):
        row_count = int(len(rows))
        if row_count <= 0:
            continue
        mode_row_counts[mode] = int(mode_row_counts.get(mode, 0) + row_count)
        mode_sequence_counts[mode] = int(mode_sequence_counts.get(mode, 0) + 1)
        symbol_row_counts[symbol] = int(symbol_row_counts.get(symbol, 0) + row_count)
        first_ts = str(rows[0].get("timestamp_utc") or "") if rows else ""
        last_ts = str(rows[-1].get("timestamp_utc") or "") if rows else ""
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
    }


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
    parser.add_argument("--seed-health-path", default=os.getenv("RUNTIME_TRAIN_SNAPSHOT_SEED_HEALTH_PATH", ""))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    rows_path = Path(args.rows_path).expanduser()
    health_path = Path(args.health_path).expanduser()
    default_seed_path = Path(DEFAULT_HEALTH_PATH).expanduser()
    seed_health_path = Path(args.seed_health_path).expanduser() if str(args.seed_health_path).strip() else None
    if seed_health_path is None and health_path != default_seed_path:
        seed_health_path = default_seed_path
    mode_allowlist = _parse_csv(args.mode_allowlist)
    symbol_allowlist = _parse_csv(args.symbol_allowlist)

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

    incremental_meta: dict[str, Any] = {}
    incremental = _incremental_snapshot_sequences(
        current_summary,
        project_root=project_root,
        health_path=health_path,
        lookback_days=max(int(args.lookback_days), 1),
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
        prefer_sqlite=bool(args.prefer_sqlite),
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
            sequences = rtc.load_runtime_observation_sequences(
                project_root,
                lookback_days=max(int(args.lookback_days), 1),
                mode_allowlist=mode_allowlist,
                symbol_allowlist=symbol_allowlist,
                prefer_sqlite=bool(args.prefer_sqlite),
                allow_snapshot=False,
            )
            incremental_meta = {"build_mode": "full_refresh"}

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
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "lookback_days": int(args.lookback_days),
        "mode_allowlist": mode_allowlist,
        "symbol_allowlist": symbol_allowlist,
        "prefer_sqlite": bool(args.prefer_sqlite),
        "rows_path": str(rows_path),
        "health_path": str(health_path),
        "jsonl_discovery_manifest": str(project_root / "governance" / "health" / "jsonl_discovery_manifest_latest.json"),
        "rows_sha256": _sha256_file(rows_path),
        "sequence_count": int(sequence_count),
        "row_count": int(row_count),
        "coverage": coverage,
    }
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
