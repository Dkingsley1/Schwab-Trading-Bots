import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _hot_jsonl_files(project_root: Path, max_files: int) -> list[Path]:
    pats = [
        "decision_explanations/**/*.jsonl",
        "decisions/**/*.jsonl",
        "governance/**/*.jsonl",
        "data/**/*.jsonl",
    ]
    files: list[Path] = []
    for pat in pats:
        files.extend(project_root.glob(pat))
    files = [p for p in files if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:max_files]


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate ingestion backlog and recommend interval scaling.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--pending-lines-threshold", type=int, default=15000)
    parser.add_argument("--interval-step-seconds", type=int, default=5)
    parser.add_argument("--max-extra-seconds", type=int, default=45)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    state_path = Path(args.state_file).resolve() if args.state_file else (project_root / "governance" / "jsonl_sql_link_state.json")

    state = {"sqlite": {}}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {"sqlite": {}}

    pending = 0
    file_count = 0
    for p in _hot_jsonl_files(project_root, args.max_files):
        rel = str(p.relative_to(project_root))
        progress = (state.get("sqlite", {}) or {}).get(rel, {})
        last_line = int(float(progress.get("last_line", 0) or 0))
        total = _safe_count_lines(p)
        if total > last_line:
            pending += (total - last_line)
            file_count += 1

    overload = pending >= args.pending_lines_threshold
    extra = min((pending // max(args.pending_lines_threshold, 1)) * args.interval_step_seconds, args.max_extra_seconds)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pending_lines": int(pending),
        "pending_files": int(file_count),
        "pending_lines_threshold": int(args.pending_lines_threshold),
        "overload": bool(overload),
        "recommended_extra_interval_seconds": int(extra if overload else 0),
    }

    out = project_root / "governance" / "health" / "ingestion_backpressure_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"ingestion_overload={payload['overload']} pending_lines={payload['pending_lines']} "
            f"pending_files={payload['pending_files']} recommended_extra_interval_seconds={payload['recommended_extra_interval_seconds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
