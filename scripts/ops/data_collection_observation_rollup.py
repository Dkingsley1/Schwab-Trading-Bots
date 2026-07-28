#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
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
LEGACY_ALIAS_BOT_ID_RE = re.compile(r"^brain_refinery_v\d+$")
STATUS_RE = re.compile(r'"status"\s*:\s*"([^"]+)"|status=([A-Z_]+)')
MAX_ARTIFACT_OBSERVATION_KEYS = 20000
DEFAULT_CHANNEL_OBSERVATION_DAYS = 7
DEFAULT_CHANNEL_TAIL_LINES = 20000


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


def _has_core_path(row: dict[str, Any]) -> bool:
    return any(
        str(row.get(key) or "").strip()
        for key in ("core_module_path", "core_file", "module_path", "core_path")
    )


def _is_collecting_row(row: dict[str, Any]) -> bool:
    return bool(
        bool(row.get("active", False))
        and bool(row.get("data_collection_active", False))
        and str(row.get("lifecycle_state") or "").strip().lower() in {"data_collection_only", "paper_live_data"}
        and _bot_id(row)
    )


def _is_bare_legacy_alias(bot_id: str) -> bool:
    return bool(LEGACY_ALIAS_BOT_ID_RE.match(str(bot_id or "").strip()))


def _has_active_canonical_sibling(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    bot_id = _bot_id(row)
    if not _is_bare_legacy_alias(bot_id):
        return False
    prefix = f"{bot_id}_"
    for candidate in rows:
        candidate_id = _bot_id(candidate)
        if (
            candidate_id.startswith(prefix)
            and _is_collecting_row(candidate)
            and _has_core_path(candidate)
        ):
            return True
    return False


def _collector_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _registry_rows(registry)
    return [
        row
        for row in rows
        if _is_collecting_row(row)
        and not _has_active_canonical_sibling(row, rows)
    ]


def _refresh_summary(payload: dict[str, Any]) -> None:
    rows = _registry_rows(payload)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    summary["total_bots"] = len(rows)
    summary["active_bots"] = sum(1 for row in rows if bool(row.get("active", False)))
    summary["data_collection_only_bots"] = sum(1 for row in rows if str(row.get("lifecycle_state") or "") == "data_collection_only")
    summary["data_collection_active_bots"] = sum(
        1 for row in rows if bool(row.get("active", False)) and bool(row.get("data_collection_active", False))
    )
    summary["training_excluded_bots"] = sum(1 for row in rows if bool(row.get("training_excluded", False)))
    summary["data_collection_training_ready_bots"] = sum(
        1
        for row in rows
        if bool(row.get("active", False))
        and bool(row.get("data_collection_active", False))
        and str(row.get("lifecycle_state") or "").strip().lower() in {"data_collection_only", "paper_live_data"}
        and bool(row.get("data_collection_training_ready", False))
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
        files.extend(root.glob(f"*/decision_explanations_{stamp}.jsonl.gz"))
    return sorted({path for path in files if path.is_file()})


def _channel_files(project_root: Path, *, days: int) -> list[Path]:
    root = project_root / "governance" / "channels"
    if not root.exists():
        return []
    files: list[Path] = []
    for stamp in _day_stamps(max(int(days), DEFAULT_CHANNEL_OBSERVATION_DAYS)):
        files.extend(root.glob(f"*/*/*_{stamp}.jsonl"))
        files.extend(root.glob(f"*/*/*_{stamp}.jsonl.gz"))
    return sorted({path for path in files if path.is_file()})


def _is_gzip_path(path: Path) -> bool:
    return path.suffix == ".gz" or path.name.endswith(".jsonl.gz")


def _read_gzip_lines(path: Path) -> list[str]:
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            return handle.readlines()
    except Exception:
        return []


def _iter_tail_lines(path: Path, *, limit: int) -> list[str]:
    max_lines = max(int(limit), 1)
    if _is_gzip_path(path):
        return _read_gzip_lines(path)[-max_lines:]

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


def _iter_new_lines(path: Path, *, offset: int, line_offset: int = 0) -> tuple[list[str], int, int]:
    try:
        size = path.stat().st_size
    except Exception:
        return [], offset, line_offset

    if _is_gzip_path(path):
        if int(offset) == size and int(line_offset) > 0:
            return [], size, int(line_offset)
        lines = _read_gzip_lines(path)
        start_line = int(line_offset) if 0 <= int(line_offset) <= len(lines) and int(offset) <= size else 0
        return lines[start_line:], size, len(lines)

    start = offset if 0 <= int(offset) <= size else 0
    out: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(start)
            out = handle.readlines()
            end = handle.tell()
    except Exception:
        return [], offset, 0
    return out, int(end), 0


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


def _iter_artifact_bot_ids(payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    direct = payload.get("added_bot_ids")
    if isinstance(direct, list):
        out.update(str(item).strip() for item in direct if str(item or "").strip())
    pack = payload.get("pack") if isinstance(payload.get("pack"), dict) else {}
    pack_ids = pack.get("bot_ids") if isinstance(pack.get("bot_ids"), list) else []
    out.update(str(item).strip() for item in pack_ids if str(item or "").strip())
    for key in ("sleeve_master_bot_id", "regression_guard_bot_id", "alpha_admission_guard_bot_id", "self_awareness_bridge_bot_id"):
        value = pack.get(key)
        if str(value or "").strip():
            out.add(str(value).strip())
    return out


def _artifact_observations(
    project_root: Path,
    *,
    bot_ids: set[str],
    seen_keys: set[str],
) -> tuple[Counter[str], dict[str, Counter[str]], list[str], int]:
    health_root = project_root / "governance" / "health"
    counts: Counter[str] = Counter()
    statuses: dict[str, Counter[str]] = defaultdict(Counter)
    new_keys: list[str] = []
    files_scanned = 0
    if not health_root.exists():
        return counts, statuses, new_keys, files_scanned

    for path in sorted(health_root.glob("*_latest.json")):
        payload = load_json(path)
        if not isinstance(payload, dict) or payload.get("ok") is False:
            continue
        artifact_bot_ids = _iter_artifact_bot_ids(payload) & bot_ids
        if not artifact_bot_ids:
            continue
        files_scanned += 1
        stamp = str(payload.get("generated_at_utc") or payload.get("timestamp_utc") or "")
        if not stamp:
            try:
                stamp = str(path.stat().st_mtime_ns)
            except Exception:
                stamp = "unknown"
        source_key = f"{path.relative_to(project_root)}:{stamp}"
        for bot_id in sorted(artifact_bot_ids):
            obs_key = f"{source_key}:{bot_id}"
            if obs_key in seen_keys:
                continue
            counts[bot_id] += 1
            statuses[bot_id]["artifact_reference"] += 1
            new_keys.append(obs_key)
    return counts, statuses, new_keys, files_scanned


def _channel_observations(
    project_root: Path,
    *,
    bot_ids: set[str],
    days: int,
    tail_lines: int,
) -> tuple[Counter[str], dict[str, Counter[str]], int, int]:
    counts: Counter[str] = Counter()
    statuses: dict[str, Counter[str]] = defaultdict(Counter)
    files_scanned = 0
    lines_scanned = 0
    for path in _channel_files(project_root, days=days):
        lines = _iter_tail_lines(path, limit=max(int(tail_lines), DEFAULT_CHANNEL_TAIL_LINES))
        file_counts, _file_statuses = _count_lines(lines, bot_ids)
        if file_counts:
            files_scanned += 1
            lines_scanned += len(lines)
        counts.update(file_counts)
        for bot_id, count in file_counts.items():
            statuses[bot_id]["channel_event"] += int(count)
    return counts, statuses, files_scanned, lines_scanned


def _diagnostic_bot_id(path: Path, payload: dict[str, Any]) -> str:
    explicit = str(payload.get("bot_id") or payload.get("id") or payload.get("name") or "").strip()
    if explicit:
        return explicit
    name = path.name
    if name.endswith("_latest.json"):
        return name[: -len("_latest.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def _diagnostic_observation_count(payload: dict[str, Any]) -> int:
    candidates = [
        payload.get("observation_count"),
        payload.get("data_collection_observations"),
        payload.get("collected_observation_count"),
        payload.get("sample_count"),
        payload.get("eligible_sequences"),
        payload.get("sequence_count"),
    ]
    for key in ("runtime_meta", "metadata", "summary"):
        nested = payload.get(key) if isinstance(payload.get(key), dict) else {}
        candidates.extend(
            [
                nested.get("observation_count"),
                nested.get("data_collection_observations"),
                nested.get("collected_observation_count"),
                nested.get("sample_count"),
                nested.get("eligible_sequences"),
                nested.get("sequence_count"),
            ]
        )
    return max([_safe_int(item, 0) for item in candidates] + [0])


def _diagnostic_observations(
    project_root: Path,
    *,
    bot_ids: set[str],
) -> tuple[Counter[str], dict[str, Counter[str]], int]:
    diagnostics_root = project_root / "governance" / "training_diagnostics"
    counts: Counter[str] = Counter()
    statuses: dict[str, Counter[str]] = defaultdict(Counter)
    files_scanned = 0
    if not diagnostics_root.exists():
        return counts, statuses, files_scanned

    for path in sorted(diagnostics_root.glob("*_latest.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        bot_id = _diagnostic_bot_id(path, payload)
        if bot_id not in bot_ids:
            continue
        count = _diagnostic_observation_count(payload)
        if count <= 0:
            continue
        files_scanned += 1
        counts[bot_id] = max(counts.get(bot_id, 0), int(count))
        status = str(payload.get("status") or payload.get("overall_status") or "training_diagnostic").strip() or "training_diagnostic"
        statuses[bot_id][f"diagnostic:{status}"] += int(count)
    return counts, statuses, files_scanned


def _collection_age_days(row: dict[str, Any]) -> float | None:
    parsed = parse_iso_utc(row.get("data_collection_started_utc"))
    if parsed is None:
        return None
    return max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0, 0.0)


def _threshold_progress(row: dict[str, Any], observations: int) -> dict[str, Any]:
    age_days = _collection_age_days(row)
    paper_standard = row.get("paper_promotion_standard") if isinstance(row.get("paper_promotion_standard"), dict) else {}
    min_observations = max(
        _safe_int(row.get("minimum_training_observations"), 0),
        _safe_int(paper_standard.get("minimum_observations"), 0),
        0,
    )
    min_days = max(
        _safe_int(row.get("minimum_data_collection_days"), 0),
        _safe_int(paper_standard.get("minimum_collection_days"), 0),
        0,
    )
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


def _row_tags(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("tags", "labeling_tags", "capability_tags", "governance_tags"):
        values = row.get(key) if isinstance(row.get(key), list) else []
        out.update(str(item).strip().lower() for item in values if str(item or "").strip())
    return out


def _is_managed_zero_observation_debt(row: dict[str, Any]) -> bool:
    bot_id = _bot_id(row).lower()
    reason = str(row.get("data_collection_reason") or row.get("training_exclusion_reason") or "").lower()
    tags = _row_tags(row)
    lifecycle_state = str(row.get("lifecycle_state") or "").strip().lower()
    mode = str(row.get("data_collection_mode") or "").strip().lower()
    training_excluded = bool(row.get("training_excluded", False) or row.get("exclude_from_training", False))
    execution_blocked = not bool(row.get("trading_enabled", False))
    is_training_labeling_observer = bool(
        "_training_labeling_" in bot_id
        or "training_labeling_intelligence" in reason
        or "collection_guard:training_labeling_intelligence_v1" in tags
        or "sleeve_family:training_labeling_intelligence" in tags
    )
    return bool(
        lifecycle_state == "data_collection_only"
        and mode in {"active_observer", "collect_only", "collection_only"}
        and training_excluded
        and execution_blocked
        and is_training_labeling_observer
    )


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
    file_line_counts = state.get("file_line_counts") if isinstance(state.get("file_line_counts"), dict) else {}
    artifact_observation_keys = [
        str(item)
        for item in (state.get("artifact_observation_keys") if isinstance(state.get("artifact_observation_keys"), list) else [])
        if str(item or "").strip()
    ]

    bootstrap = not bool(state.get("initialized", False))
    files = _decision_files(project_root, days=days)
    observed_counts: Counter[str] = Counter()
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    new_offsets: dict[str, int] = {}
    new_line_counts: dict[str, int] = {}
    files_scanned = 0
    lines_scanned = 0
    artifact_files_scanned = 0
    new_artifact_keys: list[str] = []
    channel_files_scanned = 0
    channel_lines_scanned = 0
    diagnostic_files_scanned = 0

    for path in files:
        key = str(path.relative_to(project_root))
        if bootstrap:
            lines = _iter_tail_lines(path, limit=bootstrap_tail_lines)
            try:
                new_offsets[key] = int(path.stat().st_size)
            except Exception:
                new_offsets[key] = 0
            if _is_gzip_path(path):
                new_line_counts[key] = len(_read_gzip_lines(path))
        else:
            lines, offset, line_count = _iter_new_lines(
                path,
                offset=_safe_int(file_offsets.get(key), 0),
                line_offset=_safe_int(file_line_counts.get(key), 0),
            )
            new_offsets[key] = offset
            if _is_gzip_path(path):
                new_line_counts[key] = line_count
        counts, statuses = _count_lines(lines, bot_ids)
        observed_counts.update(counts)
        for bot_id, counter in statuses.items():
            status_counts[bot_id].update(counter)
        files_scanned += 1
        lines_scanned += len(lines)

    artifact_counts, artifact_statuses, new_artifact_keys, artifact_files_scanned = _artifact_observations(
        project_root,
        bot_ids=bot_ids,
        seen_keys=set(artifact_observation_keys),
    )
    observed_counts.update(artifact_counts)
    for bot_id, counter in artifact_statuses.items():
        status_counts[bot_id].update(counter)

    channel_counts, channel_statuses, channel_files_scanned, channel_lines_scanned = _channel_observations(
        project_root,
        bot_ids=bot_ids,
        days=days,
        tail_lines=bootstrap_tail_lines,
    )
    for bot_id, counter in channel_statuses.items():
        status_counts[bot_id].update(counter)

    diagnostic_counts, diagnostic_statuses, diagnostic_files_scanned = _diagnostic_observations(
        project_root,
        bot_ids=bot_ids,
    )
    for bot_id, counter in diagnostic_statuses.items():
        status_counts[bot_id].update(counter)

    if bootstrap:
        merged_counts = Counter(
            {
                bot_id: max(
                    _safe_int(row.get("data_collection_observations"), 0),
                    observed_counts.get(bot_id, 0),
                    channel_counts.get(bot_id, 0),
                    diagnostic_counts.get(bot_id, 0),
                )
                for bot_id, row in ((_bot_id(row), row) for row in collectors)
            }
        )
    else:
        merged_counts = state_counts.copy()
        merged_counts.update(observed_counts)
        for row in collectors:
            bot_id = _bot_id(row)
            merged_counts[bot_id] = max(
                merged_counts.get(bot_id, 0),
                _safe_int(row.get("data_collection_observations"), 0),
                channel_counts.get(bot_id, 0),
                diagnostic_counts.get(bot_id, 0),
            )

    bot_updates: list[dict[str, Any]] = []
    training_ready_bot_ids: list[str] = []
    now = iso_now()
    for row in collectors:
        bot_id = _bot_id(row)
        total = int(merged_counts.get(bot_id, 0))
        progress = _threshold_progress(row, total)
        lifecycle_state = str(row.get("lifecycle_state") or "").strip().lower()
        min_observations = _safe_int(progress.get("minimum_training_observations"), 0)
        has_explicit_floor = min_observations > 0
        can_release_training_exclusion = bool(
            progress["training_ready"]
            and total > 0
            and (lifecycle_state == "data_collection_only" or has_explicit_floor)
        )
        if can_release_training_exclusion:
            training_ready_bot_ids.append(bot_id)
        desired = {
            "data_collection_observations": total,
            "collected_observation_count": total,
            "data_collection_last_counted_utc": now,
            "data_collection_observation_rollup_source": "decision_explanations_incremental",
            "data_collection_threshold_progress": progress,
            "data_collection_training_ready": can_release_training_exclusion,
        }
        if can_release_training_exclusion:
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
        elif lifecycle_state == "paper_live_data":
            desired.update(
                {
                    "training_excluded": True,
                    "exclude_from_training": True,
                    "training_exclusion_reason": (
                        "paper_live_data_requires_minimum_training_observations"
                        if not has_explicit_floor
                        else "minimum_data_collection_threshold_not_met"
                    ),
                    "training_exclusion_until": "minimum_data_collection_threshold_met",
                    "promotion_block_reason": "awaiting_data_collection_quality_gate",
                }
            )
        elif lifecycle_state == "data_collection_only" and total <= 0 and _is_managed_zero_observation_debt(row):
            managed_reason = str(row.get("training_exclusion_reason") or "collecting_training_labeling_effect_evidence_before_training")
            desired.update(
                {
                    "training_excluded": True,
                    "exclude_from_training": True,
                    "training_exclusion_reason": managed_reason,
                    "training_exclusion_until": "training_labeling_collection_threshold_met",
                    "promotion_block_reason": "awaiting_training_labeling_observation_floor",
                }
            )
        elif lifecycle_state == "data_collection_only" and total <= 0:
            desired.update(
                {
                    "training_excluded": True,
                    "exclude_from_training": True,
                    "training_exclusion_reason": "data_collection_requires_observations",
                    "training_exclusion_until": "minimum_data_collection_threshold_met",
                    "promotion_block_reason": "awaiting_data_collection_quality_gate",
                }
            )
        delta = {key: value for key, value in desired.items() if row.get(key) != value}
        if delta:
            bot_updates.append(
                {
                    "bot_id": bot_id,
                    "observations": total,
                    "training_ready": can_release_training_exclusion,
                    "threshold_training_ready": bool(progress["training_ready"]),
                    "updates": delta,
                }
            )
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
            "file_line_counts": {**file_line_counts, **new_line_counts},
            "artifact_observation_keys": list(dict.fromkeys([*artifact_observation_keys, *new_artifact_keys]))[-MAX_ARTIFACT_OBSERVATION_KEYS:],
            "last_files_scanned": [str(path.relative_to(project_root)) for path in files],
        }
        write_payload(state_path, state_payload)

    bots_with_observations = sum(1 for bot_id in bot_ids if int(merged_counts.get(bot_id, 0)) > 0)
    rows_by_bot_id = {_bot_id(row): row for row in collectors}
    raw_zero_observation_bot_ids = sorted(bot_id for bot_id in bot_ids if int(merged_counts.get(bot_id, 0)) <= 0)
    managed_zero_observation_bot_ids = sorted(
        bot_id for bot_id in raw_zero_observation_bot_ids if _is_managed_zero_observation_debt(rows_by_bot_id.get(bot_id, {}))
    )
    unmanaged_zero_observation_bot_ids = sorted(
        bot_id for bot_id in raw_zero_observation_bot_ids if bot_id not in set(managed_zero_observation_bot_ids)
    )
    training_ready = sorted(training_ready_bot_ids)
    collector_count = len(bot_ids)
    effective_bots_with_observations = min(bots_with_observations + len(managed_zero_observation_bot_ids), collector_count)
    collection_coverage_score = round((effective_bots_with_observations / max(collector_count, 1)) * 100.0, 3)
    raw_collection_coverage_score = round((bots_with_observations / max(collector_count, 1)) * 100.0, 3)
    zero_observation_penalty = round((len(unmanaged_zero_observation_bot_ids) / max(collector_count, 1)) * 100.0, 3)
    raw_zero_observation_penalty = round((len(raw_zero_observation_bot_ids) / max(collector_count, 1)) * 100.0, 3)
    data_quality_score = round(max(0.0, min(collection_coverage_score - zero_observation_penalty, 100.0)), 3)
    raw_data_quality_score = round(max(0.0, min(raw_collection_coverage_score - raw_zero_observation_penalty, 100.0)), 3)
    training_readiness_score = round((len(training_ready) / max(collector_count, 1)) * 100.0, 3)
    zero_repair_commands = [
        ["./scripts/ops/opsctl.sh", "training-label-audit", "--json"],
        ["./scripts/ops/opsctl.sh", "data-collection-observation-rollup", "--bootstrap-tail-lines", "5000", "--json"],
    ]
    if unmanaged_zero_observation_bot_ids:
        zero_repair_commands.insert(
            0,
            [
                "./scripts/ops/opsctl.sh",
                "bot-needs",
                "--include-bot-ids",
                ",".join(unmanaged_zero_observation_bot_ids[:12]),
                "--json",
            ],
        )
    ok = len(unmanaged_zero_observation_bot_ids) == 0
    return {
        "timestamp_utc": now,
        "schema_version": 1,
        "ok": ok,
        "overall_status": "ready" if ok else "degraded",
        "mode": "bootstrap_tail" if bootstrap else "incremental",
        "apply": bool(apply),
        "collector_count": collector_count,
        "bots_with_observations": bots_with_observations,
        "effective_bots_with_observations": effective_bots_with_observations,
        "zero_observation_count": len(unmanaged_zero_observation_bot_ids),
        "zero_observation_bot_ids": unmanaged_zero_observation_bot_ids[:50],
        "unmanaged_zero_observation_count": len(unmanaged_zero_observation_bot_ids),
        "unmanaged_zero_observation_bot_ids": unmanaged_zero_observation_bot_ids[:50],
        "managed_zero_observation_count": len(managed_zero_observation_bot_ids),
        "managed_zero_observation_bot_ids": managed_zero_observation_bot_ids[:50],
        "raw_zero_observation_count": len(raw_zero_observation_bot_ids),
        "raw_zero_observation_bot_ids": raw_zero_observation_bot_ids[:50],
        "data_quality_score": data_quality_score,
        "raw_data_quality_score": raw_data_quality_score,
        "collection_coverage_score": collection_coverage_score,
        "raw_collection_coverage_score": raw_collection_coverage_score,
        "training_readiness_score": training_readiness_score,
        "quality_contract": {
            "data_quality_definition": "observation coverage for active collection bots; training readiness is reported separately",
            "managed_zero_definition": "collect-only training-labeling observers with execution blocked and explicit training exclusion are managed observer debt, not paper-trading collection failures",
            "target_data_quality_score": 100.0,
            "target_training_readiness_score": 100.0,
            "data_quality_score": data_quality_score,
            "raw_data_quality_score": raw_data_quality_score,
            "collection_coverage_score": collection_coverage_score,
            "raw_collection_coverage_score": raw_collection_coverage_score,
            "training_readiness_score": training_readiness_score,
            "training_ready_count": len(training_ready),
            "training_ready_gap": max(collector_count - len(training_ready), 0),
            "managed_zero_observation_count": len(managed_zero_observation_bot_ids),
            "unmanaged_zero_observation_count": len(unmanaged_zero_observation_bot_ids),
        },
        "zero_observation_repair_lane": {
            "active": bool(unmanaged_zero_observation_bot_ids),
            "zero_observation_count": len(unmanaged_zero_observation_bot_ids),
            "target_bot_ids": unmanaged_zero_observation_bot_ids[:12],
            "priority": "targeted" if unmanaged_zero_observation_bot_ids else "observe",
            "recommended_commands": zero_repair_commands if unmanaged_zero_observation_bot_ids else [],
            "policy": "repair_exact_zero_observation_bots_before_broad_collection_expansion",
        },
        "managed_zero_observation_lane": {
            "active": bool(managed_zero_observation_bot_ids),
            "managed_count": len(managed_zero_observation_bot_ids),
            "target_bot_ids": managed_zero_observation_bot_ids[:12],
            "recommended_commands": [
                [
                    "./scripts/ops/opsctl.sh",
                    "training-labeling-intelligence",
                    "--apply",
                    "--materialize-collect-only-diagnostics",
                    "--json",
                ]
            ]
            if managed_zero_observation_bot_ids
            else [],
            "policy": "keep expected collect-only training-labeling observers excluded from training and execution until observation and minimum-day gates clear",
        },
        "files_scanned": files_scanned,
        "lines_scanned": lines_scanned,
        "artifact_files_scanned": artifact_files_scanned,
        "new_artifact_observations_counted": len(new_artifact_keys),
        "channel_files_scanned": channel_files_scanned,
        "channel_lines_scanned": channel_lines_scanned,
        "channel_observations_counted": int(sum(channel_counts.values())),
        "diagnostic_files_scanned": diagnostic_files_scanned,
        "diagnostic_observations_counted": int(sum(diagnostic_counts.values())),
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
            "repair exact zero-observation bot routing before adding broad collection volume" if unmanaged_zero_observation_bot_ids else "if unmanaged_zero_observation_count rises above zero, inspect decision and channel routing before adding more bots",
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
