import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from link_jsonl_to_sql import discover_jsonl_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IGNORED_BACKPRESSURE_PREFIXES = (
    "governance/health/jsonl_ingest_batch_journal",
    "governance/events/jsonl_ingest_batches_",
)
DEFERRED_BACKPRESSURE_PREFIXES = (
    "decision_explanations/",
    "governance/events/api_calls_",
    "governance/events/data_ingress_",
    "governance/events/gate_logs_",
    "governance/events/loop_state_",
    "governance/channels/api/",
    "governance/channels/gate/",
    "governance/channels/ingress/",
    "governance/channels/loop_state/",
    "governance/channels/risk/",
    "governance/channels/runtime/",
)
IGNORED_BACKPRESSURE_SUFFIXES = (
    "/runtime_telemetry.jsonl",
)
DEFERRED_BACKPRESSURE_CONTAINS = (
    "/shadow_pnl_attribution_",
)


def _safe_count_lines(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _estimated_total_lines(path: Path, stat, progress: dict, *, max_exact_bytes: int, sample_bytes: int) -> int:
    size_bytes = int(stat.st_size)
    if size_bytes <= max(int(max_exact_bytes), 0):
        return _safe_count_lines(path)

    last_line = int(float(progress.get("last_line", 0) or 0))
    prev_size = int(float(progress.get("file_size_bytes", 0) or 0))
    if last_line > 0 and prev_size > 0:
        # Reuse prior ingestion density to avoid rescanning multi-GB files on every verify.
        est = int(round((size_bytes / max(prev_size, 1)) * last_line))
        return max(est, last_line)

    sample_target = min(max(int(sample_bytes), 4096), size_bytes)
    try:
        with path.open("rb") as f:
            sample = f.read(sample_target)
    except Exception:
        sample = b""

    if sample:
        newline_count = sample.count(b"\n")
        if newline_count > 0:
            avg_bytes_per_line = max(len(sample) / newline_count, 1.0)
            est = int(round(size_bytes / avg_bytes_per_line))
            return max(est, 1)

    # Conservative fallback when we cannot sample content.
    return max(size_bytes // 256, 1)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _last_line_for_state(rel: str, stat, progress: dict) -> int:
    last_line = int(float(progress.get("last_line", 0) or 0))
    prev_mtime = float(progress.get("mtime", 0.0) or 0.0)
    prev_inode = int(float(progress.get("file_inode", 0) or 0))
    prev_size = int(float(progress.get("file_size_bytes", 0) or 0))

    if prev_inode > 0 and int(stat.st_ino) != prev_inode:
        return 0
    if prev_size > 0 and int(stat.st_size) < prev_size:
        return 0
    if float(stat.st_mtime) < prev_mtime:
        return 0
    return max(last_line, 0)


def _progress_sort_key(progress: dict) -> tuple[int, int, float, int]:
    return (
        int(float(progress.get("last_line", 0) or 0)),
        int(float(progress.get("file_size_bytes", 0) or 0)),
        float(progress.get("mtime", 0.0) or 0.0),
        int(float(progress.get("last_offset_bytes", 0) or 0)),
    )


def _merge_sqlite_progress(merged: dict[str, dict], entries: dict[str, dict]) -> None:
    for rel, raw_progress in entries.items():
        progress = raw_progress if isinstance(raw_progress, dict) else {}
        current = merged.get(str(rel), {})
        if not isinstance(current, dict) or _progress_sort_key(progress) >= _progress_sort_key(current):
            merged[str(rel)] = progress


def _load_sqlite_progress(path: Path) -> dict[str, dict]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {}
    sqlite_state = payload.get("sqlite", {})
    return sqlite_state if isinstance(sqlite_state, dict) else {}


def _resolve_sqlite_state(project_root: Path, state_file: str | None) -> tuple[dict[str, dict], list[str], str]:
    if state_file:
        state_path = Path(state_file).resolve()
        return _load_sqlite_progress(state_path), [str(state_path)], "explicit"

    shard_root = project_root / "governance" / "sql_link_shards"
    shard_files = sorted(p for p in shard_root.glob("jsonl_sql_link_state_*.json") if p.is_file())
    legacy_path = project_root / "governance" / "jsonl_sql_link_state.json"

    state_files: list[Path] = []
    if shard_files:
        state_files.extend(shard_files)
    if legacy_path.exists():
        state_files.append(legacy_path)
    if not state_files:
        return {}, [], "missing"

    merged: dict[str, dict] = {}
    for path in state_files:
        _merge_sqlite_progress(merged, _load_sqlite_progress(path))

    mode = "sharded_merged" if shard_files else "legacy"
    return merged, [str(path) for path in state_files], mode


def _record_top_pending(rows: list[dict], *, rel: str, pending: int, age_seconds: float, total: int, last_line: int, top_n: int) -> None:
    if pending <= 0:
        return
    rows.append(
        {
            "source_rel": str(rel),
            "pending_lines": int(pending),
            "oldest_pending_age_seconds": round(float(age_seconds), 3),
            "total_lines": int(total),
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


def _should_ignore_backpressure_file(rel: str) -> bool:
    normalized = str(rel or "")
    return any(normalized.startswith(prefix) for prefix in IGNORED_BACKPRESSURE_PREFIXES) or any(
        normalized.endswith(suffix) for suffix in IGNORED_BACKPRESSURE_SUFFIXES
    )


def _is_deferred_backpressure_file(rel: str) -> bool:
    normalized = str(rel or "")
    if _should_ignore_backpressure_file(normalized):
        return False
    return any(normalized.startswith(prefix) for prefix in DEFERRED_BACKPRESSURE_PREFIXES) or any(
        token in normalized for token in DEFERRED_BACKPRESSURE_CONTAINS
    )


def _age_pressure_triggered(
    *,
    oldest_pending_age_seconds: float,
    pending_lines: int,
    threshold_seconds: float,
    min_pending_lines: int,
) -> bool:
    return bool(
        float(oldest_pending_age_seconds) >= float(threshold_seconds)
        and int(pending_lines) >= max(int(min_pending_lines), 1)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate ingestion backlog and recommend interval scaling.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-exact-count-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--sample-bytes", type=int, default=256 * 1024)
    parser.add_argument("--pending-lines-threshold", type=int, default=15000)
    parser.add_argument("--pending-files-threshold", type=int, default=45)
    parser.add_argument("--oldest-age-threshold-seconds", type=int, default=240)
    parser.add_argument("--oldest-age-min-pending-lines", type=int, default=100)
    parser.add_argument("--oldest-age-min-file-pending-lines", type=int, default=100)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--trend-ratio-threshold", type=float, default=1.20)
    parser.add_argument("--trend-min-delta-lines", type=int, default=500)
    parser.add_argument("--interval-step-seconds", type=int, default=5)
    parser.add_argument("--max-extra-seconds", type=int, default=60)
    parser.add_argument("--top-pending-files", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    sqlite_state, state_files, state_mode = _resolve_sqlite_state(project_root, args.state_file)

    files = discover_jsonl_files(project_root)
    if args.max_files > 0:
        files = files[: int(args.max_files)]

    pending_core = 0
    file_count_core = 0
    oldest_pending_age_seconds_core = 0.0
    top_pending_files_core: list[dict] = []
    pending_deferred = 0
    file_count_deferred = 0
    oldest_pending_age_seconds_deferred = 0.0
    top_pending_files_deferred: list[dict] = []

    now_ts = datetime.now(timezone.utc).timestamp()
    for p in files:
        try:
            rel = str(p.relative_to(project_root))
            st = p.stat()
        except Exception:
            continue

        if _should_ignore_backpressure_file(rel):
            continue

        progress = sqlite_state.get(rel, {}) if isinstance(sqlite_state.get(rel, {}), dict) else {}
        last_line = _last_line_for_state(rel, st, progress)
        total = _estimated_total_lines(
            p,
            st,
            progress,
            max_exact_bytes=int(args.max_exact_count_bytes),
            sample_bytes=int(args.sample_bytes),
        )
        pending_lines = max(int(total) - int(last_line), 0)
        if pending_lines <= 0:
            continue

        age_seconds = max(float(now_ts) - float(st.st_mtime), 0.0)
        if _is_deferred_backpressure_file(rel):
            file_count_deferred += 1
            pending_deferred += pending_lines
            if pending_lines >= max(int(args.oldest_age_min_file_pending_lines), 1):
                oldest_pending_age_seconds_deferred = max(oldest_pending_age_seconds_deferred, age_seconds)
            _record_top_pending(
                top_pending_files_deferred,
                rel=rel,
                pending=pending_lines,
                age_seconds=age_seconds,
                total=total,
                last_line=last_line,
                top_n=max(int(args.top_pending_files), 1),
            )
        else:
            file_count_core += 1
            pending_core += pending_lines
            if pending_lines >= max(int(args.oldest_age_min_file_pending_lines), 1):
                oldest_pending_age_seconds_core = max(oldest_pending_age_seconds_core, age_seconds)
            _record_top_pending(
                top_pending_files_core,
                rel=rel,
                pending=pending_lines,
                age_seconds=age_seconds,
                total=total,
                last_line=last_line,
                top_n=max(int(args.top_pending_files), 1),
            )

    out = project_root / "governance" / "health" / "ingestion_backpressure_latest.json"
    prev = _load_json(out)

    alpha = min(max(float(args.ema_alpha), 0.01), 1.0)
    prev_ema = float(prev.get("ema_pending_lines", prev.get("pending_lines", 0.0)) or 0.0)
    prev_pending = int(prev.get("pending_lines", 0) or 0)
    ema_pending = (alpha * float(pending_core)) + ((1.0 - alpha) * prev_ema)
    delta = int(pending_core) - int(prev_pending)

    trend_floor = max(int(float(prev_pending) * max(float(args.trend_ratio_threshold), 1.0)), prev_pending + max(int(args.trend_min_delta_lines), 0))
    trend_up = bool(pending_core >= trend_floor and pending_core > 0)

    meaningful_pending_for_pressure = max(int(float(args.pending_lines_threshold) * 0.75), int(args.oldest_age_min_pending_lines))
    line_pressure = bool(pending_core >= int(args.pending_lines_threshold))
    file_pressure = bool(
        file_count_core >= int(args.pending_files_threshold)
        and pending_core >= meaningful_pending_for_pressure
    )
    age_pressure = _age_pressure_triggered(
        oldest_pending_age_seconds=oldest_pending_age_seconds_core,
        pending_lines=pending_core,
        threshold_seconds=float(args.oldest_age_threshold_seconds),
        min_pending_lines=int(args.oldest_age_min_pending_lines),
    )
    ema_pressure = bool(
        ema_pending >= max(float(args.pending_lines_threshold) * 0.8, 1.0)
        and trend_up
        and pending_core >= meaningful_pending_for_pressure
    )

    overload = bool(age_pressure or line_pressure or (file_pressure and trend_up) or ema_pressure)

    if overload:
        line_ratio = float(pending_core) / max(float(args.pending_lines_threshold), 1.0)
        age_ratio = float(oldest_pending_age_seconds_core) / max(float(args.oldest_age_threshold_seconds), 1.0)
        ema_ratio = float(ema_pending) / max(float(args.pending_lines_threshold), 1.0)
        severity = max(line_ratio, age_ratio, ema_ratio, 1.0)
        steps = max(int(round((severity - 1.0) * 2.0)) + 1, 1)
        extra = min(steps * max(int(args.interval_step_seconds), 1), max(int(args.max_extra_seconds), 0))
    else:
        extra = 0

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files_scanned": int(len(files)),
        "state_file": state_files[0] if state_files else "",
        "state_files": list(state_files),
        "state_mode": state_mode,
        "pending_lines": int(pending_core),
        "pending_files": int(file_count_core),
        "pending_lines_total": int(pending_core + pending_deferred),
        "pending_files_total": int(file_count_core + file_count_deferred),
        "pending_lines_deferred": int(pending_deferred),
        "pending_files_deferred": int(file_count_deferred),
        "pending_lines_threshold": int(args.pending_lines_threshold),
        "pending_files_threshold": int(args.pending_files_threshold),
        "meaningful_pending_for_file_pressure": int(meaningful_pending_for_pressure),
        "oldest_pending_age_seconds": round(float(oldest_pending_age_seconds_core), 3),
        "oldest_pending_age_seconds_total": round(
            float(max(oldest_pending_age_seconds_core, oldest_pending_age_seconds_deferred)), 3
        ),
        "oldest_pending_age_seconds_deferred": round(float(oldest_pending_age_seconds_deferred), 3),
        "oldest_age_threshold_seconds": int(args.oldest_age_threshold_seconds),
        "oldest_age_min_pending_lines": int(args.oldest_age_min_pending_lines),
        "oldest_age_min_file_pending_lines": int(args.oldest_age_min_file_pending_lines),
        "ema_pending_lines": round(float(ema_pending), 3),
        "ema_alpha": float(alpha),
        "pending_lines_delta": int(delta),
        "trend_up": bool(trend_up),
        "line_pressure": bool(line_pressure),
        "file_pressure": bool(file_pressure),
        "age_pressure": bool(age_pressure),
        "ema_pressure": bool(ema_pressure),
        "overload": bool(overload),
        "recommended_extra_interval_seconds": int(extra),
        "deferred_backpressure_classes": [
            "governance/events/api_calls_*",
            "governance/events/data_ingress_*",
            "governance/events/loop_state_*",
            "governance/channels/{api,ingress,loop_state,runtime}/*",
            "governance/shadow_*/shadow_pnl_attribution_*",
        ],
        "top_pending_files": top_pending_files_core,
        "top_deferred_pending_files": top_pending_files_deferred,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"ingestion_overload={payload['overload']} pending_lines={payload['pending_lines']} "
            f"pending_files={payload['pending_files']} oldest_pending_age_s={payload['oldest_pending_age_seconds']:.1f} "
            f"ema_pending_lines={payload['ema_pending_lines']:.1f} "
            f"recommended_extra_interval_seconds={payload['recommended_extra_interval_seconds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
