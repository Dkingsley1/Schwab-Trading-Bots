#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, parse_iso_utc, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, parse_iso_utc, write_payload


DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "data_collection_observation_rollup_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "data_collection_observation_rollup_state.json"
BOT_ID_RE = re.compile(r"brain_refinery_v\d+_[A-Za-z0-9_]+")
STATUS_RE = re.compile(r'"status"\s*:\s*"([^"]+)"|status=([A-Z_]+)')


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _registry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _bot_id(row: dict[str, Any]) -> str:
    return str(row.get("bot_id") or row.get("id") or row.get("name") or "").strip()


def _collector_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in _registry_rows(registry)
        if bool(row.get("active", False))
        and bool(row.get("data_collection_active", False))
        and str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only"
        and _bot_id(row)
    ]


def _refresh_summary(payload: dict[str, Any]) -> None:
    rows = _registry_rows(payload)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    summary["total_bots"] = len(rows)
    summary["active_bots"] = sum(1 for row in rows if bool(row.get("active", False)))
    summary["data_collection_only_bots"] = sum(1 for row in rows if str(row.get("lifecycle_state") or "") == "data_collection_only")
    summary["training_excluded_bots"] = sum(1 for row in rows if bool(row.get("training_excluded", False)))
    summary["data_collection_training_ready_bots"] = sum(
        1 for row in rows if str(row.get("lifecycle_state") or "") == "data_collection_only" and bool(row.get("data_collection_training_ready", False))
    )
    payload["summary"] = summary
    payload["updated_at_utc"] = iso_now()


def _day_stamps(days: int) -> list[str]:
    now = datetime.now(timezone.utc)
    return [(now - timedelta(days=offset)).strftime("%Y%m%d") for offset in range(max(int(days), 1))]


def _decision_files(project_root: Path, *, days: int) -> list[Path]:
    root = project_root / "decision_explanations"
    if not root.exists():
        return []
    files: list[Path] = []
    for stamp in _day_stamps(days):
        files.extend(root.glob(f"*/decision_explanations_{stamp}.jsonl"))
    return sorted({path for path in files if path.is_file()})


def _iter_tail_lines(path: Path, *, limit: int) -> list[str]:
    max_lines = max(int(limit), 1)
    block_size = 64 * 1024
    chunks: deque[bytes] = deque()
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            newline_count = 0
            while position > 0 and newline_count <= max_lines:
                read_size = min(block_size, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                chunks.appendleft(chunk)
                newline_count += chunk.count(b"\n")
    except Exception:
        return []
    text = b"".join(chunks).decode("utf-8", errors="ignore")
    return text.splitlines(keepends=True)[-max_lines:]


def _iter_new_lines(path: Path, *, offset: int) -> tuple[list[str], int]:
    try:
        size = path.stat().st_size
    except Exception:
        return [], offset
    start = offset if 0 <= int(offset) <= size else 0
    out: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(start)
            out = handle.readlines()
            end = handle.tell()
    except Exception:
        return [], offset
    return out, int(end)


def _count_lines(lines: list[str], bot_ids: set[str]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    counts: Counter[str] = Counter()
    statuses: dict[str, Counter[str]] = defaultdict(Counter)
    for line in lines:
        matches = [match for match in BOT_ID_RE.findall(line) if match in bot_ids]
        if not matches:
            continue
        bot_id = matches[0]
        counts[bot_id] += 1
        status_match = STATUS_RE.search(line)
        status = "unknown"
        if status_match:
            status = str(status_match.group(1) or status_match.group(2) or "unknown")
        statuses[bot_id][status] += 1
    return counts, statuses


def _collection_age_days(row: dict[str, Any]) -> float | None:
    parsed = parse_iso_utc(row.get("data_collection_started_utc"))
    if parsed is None:
        return None
    return max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0, 0.0)


def _threshold_progress(row: dict[str, Any], observations: int) -> dict[str, Any]:
    age_days = _collection_age_days(row)
    min_observations = max(_safe_int(row.get("minimum_training_observations"), 0), 0)
    min_days = max(_safe_int(row.get("minimum_data_collection_days"), 0), 0)
    observations_ready = bool(min_observations <= 0 or observations >= min_observations)
    days_ready = bool(min_days <= 0 or (age_days is not None and age_days >= float(min_days)))
    return {
        "observations": int(observations),
        "minimum_training_observations": min_observations,
        "observations_ready": observations_ready,
        "collection_age_days": round(float(age_days), 3) if age_days is not None else None,
        "minimum_data_collection_days": min_days,
        "days_ready": days_ready,
        "training_ready": bool(observations_ready and days_ready),
    }


def build_payload(
    *,
    project_root: Path,
    registry_path: Path,
    state_path: Path,
    days: int,
    bootstrap_tail_lines: int,
    apply: bool,
) -> dict[str, Any]:
    registry = load_json(registry_path)
    collectors = _collector_rows(registry)
    bot_ids = {_bot_id(row) for row in collectors}
    state = load_json(state_path)
    state_counts = Counter({str(k): _safe_int(v, 0) for k, v in (state.get("counts") if isinstance(state.get("counts"), dict) else {}).items()})
    file_offsets = state.get("file_offsets") if isinstance(state.get("file_offsets"), dict) else {}

    bootstrap = not bool(state.get("initialized", False))
    files = _decision_files(project_root, days=days)
    observed_counts: Counter[str] = Counter()
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    new_offsets: dict[str, int] = {}
    files_scanned = 0
    lines_scanned = 0

    for path in files:
        key = str(path.relative_to(project_root))
        if bootstrap:
            lines = _iter_tail_lines(path, limit=bootstrap_tail_lines)
            try:
                new_offsets[key] = int(path.stat().st_size)
            except Exception:
                new_offsets[key] = 0
        else:
            lines, offset = _iter_new_lines(path, offset=_safe_int(file_offsets.get(key), 0))
            new_offsets[key] = offset
        counts, statuses = _count_lines(lines, bot_ids)
        observed_counts.update(counts)
        for bot_id, counter in statuses.items():
            status_counts[bot_id].update(counter)
        files_scanned += 1
        lines_scanned += len(lines)

    if bootstrap:
        merged_counts = Counter({bot_id: max(_safe_int(row.get("data_collection_observations"), 0), observed_counts.get(bot_id, 0)) for bot_id, row in ((_bot_id(row), row) for row in collectors)})
    else:
        merged_counts = state_counts.copy()
        merged_counts.update(observed_counts)
        for row in collectors:
            bot_id = _bot_id(row)
            merged_counts[bot_id] = max(merged_counts.get(bot_id, 0), _safe_int(row.get("data_collection_observations"), 0))

    bot_updates: list[dict[str, Any]] = []
    now = iso_now()
    for row in collectors:
        bot_id = _bot_id(row)
        total = int(merged_counts.get(bot_id, 0))
        progress = _threshold_progress(row, total)
        desired = {
            "data_collection_observations": total,
            "collected_observation_count": total,
            "data_collection_last_counted_utc": now,
            "data_collection_observation_rollup_source": "decision_explanations_incremental",
            "data_collection_threshold_progress": progress,
            "data_collection_training_ready": bool(progress["training_ready"]),
        }
        if progress["training_ready"]:
            desired.update(
                {
                    "training_excluded": False,
                    "exclude_from_training": False,
                    "training_exclusion_reason": "",
                    "training_exclusion_until": "",
                    "promotion_blocked_until": "",
                    "promotion_block_reason": "awaiting_training_quality_gate",
                }
            )
        delta = {key: value for key, value in desired.items() if row.get(key) != value}
        if delta:
            bot_updates.append({"bot_id": bot_id, "observations": total, "training_ready": bool(progress["training_ready"]), "updates": delta})
            if apply:
                row.update(delta)

    if apply:
        _refresh_summary(registry)
        registry_path.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")
        state_payload = {
            "timestamp_utc": now,
            "initialized": True,
            "counts": {bot_id: int(merged_counts.get(bot_id, 0)) for bot_id in sorted(bot_ids)},
            "file_offsets": {**file_offsets, **new_offsets},
            "last_files_scanned": [str(path.relative_to(project_root)) for path in files],
        }
        write_payload(state_path, state_payload)

    bots_with_observations = sum(1 for bot_id in bot_ids if int(merged_counts.get(bot_id, 0)) > 0)
    training_ready = [row["bot_id"] for row in bot_updates if bool(row.get("training_ready", False))]
    return {
        "timestamp_utc": now,
        "schema_version": 1,
        "ok": bots_with_observations == len(bot_ids),
        "overall_status": "ready" if bots_with_observations == len(bot_ids) else "degraded",
        "mode": "bootstrap_tail" if bootstrap else "incremental",
        "apply": bool(apply),
        "collector_count": len(bot_ids),
        "bots_with_observations": bots_with_observations,
        "zero_observation_count": max(len(bot_ids) - bots_with_observations, 0),
        "files_scanned": files_scanned,
        "lines_scanned": lines_scanned,
        "new_rows_counted": int(sum(observed_counts.values())),
        "total_observations": int(sum(merged_counts.get(bot_id, 0) for bot_id in bot_ids)),
        "training_ready_count": len(training_ready),
        "training_ready_bot_ids": training_ready[:25],
        "top_collectors": [
            {"bot_id": bot_id, "observations": int(count)}
            for bot_id, count in sorted(((bot_id, merged_counts.get(bot_id, 0)) for bot_id in bot_ids), key=lambda item: int(item[1]), reverse=True)[:20]
        ],
        "status_counts_top": {bot_id: dict(status_counts.get(bot_id, {})) for bot_id, _ in observed_counts.most_common(10)},
        "updated_bot_count": len(bot_updates),
        "bot_updates": bot_updates[:50],
        "state_path": str(state_path),
        "registry_path": str(registry_path),
        "recommended_actions": [
            "run this rollup on a short cadence so training thresholds use registry counters instead of ad hoc log scans",
            "keep data-only bots excluded from training until both observation and minimum-day gates clear",
            "if zero_observation_count rises above zero, inspect decision_explanations routing before adding more bots",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally roll data-only bot observations from decision explanations back into the registry.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--bootstrap-tail-lines", type=int, default=1200)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root=project_root,
        registry_path=Path(args.registry).expanduser(),
        state_path=Path(args.state_file).expanduser(),
        days=args.days,
        bootstrap_tail_lines=args.bootstrap_tail_lines,
        apply=bool(args.apply),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "data_collection_observation_rollup "
            f"overall_status={payload.get('overall_status')} "
            f"collector_count={payload.get('collector_count')} "
            f"bots_with_observations={payload.get('bots_with_observations')} "
            f"total_observations={payload.get('total_observations')}"
        )
    return 0 if payload.get("overall_status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
