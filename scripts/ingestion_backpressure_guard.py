import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from link_jsonl_to_sql import discover_jsonl_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--trend-ratio-threshold", type=float, default=1.20)
    parser.add_argument("--trend-min-delta-lines", type=int, default=500)
    parser.add_argument("--interval-step-seconds", type=int, default=5)
    parser.add_argument("--max-extra-seconds", type=int, default=60)
    parser.add_argument("--top-pending-files", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    state_path = Path(args.state_file).resolve() if args.state_file else (project_root / "governance" / "jsonl_sql_link_state.json")

    state = {"sqlite": {}}
    if state_path.exists():
        state = _load_json(state_path)
        if not isinstance(state, dict):
            state = {"sqlite": {}}
    sqlite_state = state.get("sqlite", {}) if isinstance(state.get("sqlite", {}), dict) else {}

    files = discover_jsonl_files(project_root)
    if args.max_files > 0:
        files = files[: int(args.max_files)]

    pending = 0
    file_count = 0
    oldest_pending_age_seconds = 0.0
    top_pending_files: list[dict] = []

    now_ts = datetime.now(timezone.utc).timestamp()
    for p in files:
        try:
            rel = str(p.relative_to(project_root))
            st = p.stat()
        except Exception:
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

        file_count += 1
        pending += pending_lines
        age_seconds = max(float(now_ts) - float(st.st_mtime), 0.0)
        oldest_pending_age_seconds = max(oldest_pending_age_seconds, age_seconds)
        _record_top_pending(
            top_pending_files,
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
    ema_pending = (alpha * float(pending)) + ((1.0 - alpha) * prev_ema)
    delta = int(pending) - int(prev_pending)

    trend_floor = max(int(float(prev_pending) * max(float(args.trend_ratio_threshold), 1.0)), prev_pending + max(int(args.trend_min_delta_lines), 0))
    trend_up = bool(pending >= trend_floor and pending > 0)

    line_pressure = bool(pending >= int(args.pending_lines_threshold))
    file_pressure = bool(file_count >= int(args.pending_files_threshold))
    age_pressure = bool(oldest_pending_age_seconds >= float(args.oldest_age_threshold_seconds))
    ema_pressure = bool(ema_pending >= max(float(args.pending_lines_threshold) * 0.8, 1.0) and trend_up)

    overload = bool(age_pressure or line_pressure or (file_pressure and trend_up) or ema_pressure)

    if overload:
        line_ratio = float(pending) / max(float(args.pending_lines_threshold), 1.0)
        age_ratio = float(oldest_pending_age_seconds) / max(float(args.oldest_age_threshold_seconds), 1.0)
        ema_ratio = float(ema_pending) / max(float(args.pending_lines_threshold), 1.0)
        severity = max(line_ratio, age_ratio, ema_ratio, 1.0)
        steps = max(int(round((severity - 1.0) * 2.0)) + 1, 1)
        extra = min(steps * max(int(args.interval_step_seconds), 1), max(int(args.max_extra_seconds), 0))
    else:
        extra = 0

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files_scanned": int(len(files)),
        "state_file": str(state_path),
        "pending_lines": int(pending),
        "pending_files": int(file_count),
        "pending_lines_threshold": int(args.pending_lines_threshold),
        "pending_files_threshold": int(args.pending_files_threshold),
        "oldest_pending_age_seconds": round(float(oldest_pending_age_seconds), 3),
        "oldest_age_threshold_seconds": int(args.oldest_age_threshold_seconds),
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
        "top_pending_files": top_pending_files,
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
