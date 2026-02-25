import argparse
import json
from pathlib import Path

from snapshot_health_sql import load_snapshot_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync snapshot health payloads into SQLite for retrain-time SQL fallback.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--sqlite-path", default="")
    parser.add_argument("--prefer-sql", action="store_true")
    parser.add_argument("--prefer-files", action="store_true")
    parser.add_argument("--no-persist", action="store_true")
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

    out = {
        "timestamp_utc": meta.get("coverage_ts") or meta.get("replay_ts") or "",
        "project_root": str(project_root),
        "sqlite_path": meta.get("sqlite_path"),
        "source_mode": meta.get("source_mode"),
        "selected_source": meta.get("selected_source"),
        "sql_sync": meta.get("sql_sync"),
        "context": {k: round(float(v), 6) for k, v in (context or {}).items()},
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(json.dumps(out, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
