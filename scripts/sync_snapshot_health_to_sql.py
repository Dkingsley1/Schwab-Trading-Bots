import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from snapshot_health_sql import debug_snapshot_ingest_coverage, load_snapshot_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync snapshot health payloads into SQLite for retrain-time SQL fallback.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--sqlite-path", default="")
    parser.add_argument("--prefer-sql", action="store_true")
    parser.add_argument("--prefer-files", action="store_true")
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument(
        "--require-full-debug-sync",
        action="store_true",
        default=False,
        help="Exit non-zero unless all retained debug snapshot dirs are fully ingested into SQLite.",
    )
    parser.add_argument(
        "--min-debug-sync-ratio",
        type=float,
        default=1.0,
        help="Minimum debug snapshot ingest coverage ratio required when --require-full-debug-sync is set.",
    )
    parser.add_argument(
        "--out-file",
        default="",
        help="Optional health artifact path. Defaults to governance/health/snapshot_sql_sync_latest.json.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    sqlite_path = Path(args.sqlite_path).expanduser().resolve() if str(args.sqlite_path).strip() else None
    prefer_sql = True
    if args.prefer_files:
        prefer_sql = False
    elif args.prefer_sql:
        prefer_sql = True

    context, meta = load_snapshot_context(
        project_root=project_root,
        sqlite_path=sqlite_path,
        prefer_sql=prefer_sql,
        persist_files_to_sql=not bool(args.no_persist),
    )

    debug_coverage = debug_snapshot_ingest_coverage(
        project_root=project_root,
        sqlite_path=Path(meta.get("sqlite_path")).expanduser().resolve() if str(meta.get("sqlite_path") or "").strip() else sqlite_path,
    )
    required_ratio = max(min(float(args.min_debug_sync_ratio), 1.0), 0.0)
    debug_sync_ok = bool(debug_coverage.get("all_ready", False)) and float(debug_coverage.get("coverage_ratio", 0.0) or 0.0) >= required_ratio

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "sqlite_path": meta.get("sqlite_path"),
        "source_mode": meta.get("source_mode"),
        "selected_source": meta.get("selected_source"),
        "sql_sync": meta.get("sql_sync"),
        "raw_debug_sync": meta.get("raw_debug_sync"),
        "raw_debug_context": meta.get("raw_debug_context"),
        "raw_context_enabled": bool(meta.get("raw_context_enabled", False)),
        "debug_snapshot_ingest_coverage": debug_coverage,
        "require_full_debug_sync": bool(args.require_full_debug_sync),
        "min_debug_sync_ratio": float(required_ratio),
        "debug_sync_ok": bool(debug_sync_ok),
        "context": {k: round(float(v), 6) for k, v in (context or {}).items()},
    }

    out_path = Path(args.out_file).expanduser().resolve() if str(args.out_file).strip() else (project_root / "governance" / "health" / "snapshot_sql_sync_latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(json.dumps(out, ensure_ascii=True, indent=2))
    if args.require_full_debug_sync and not debug_sync_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
