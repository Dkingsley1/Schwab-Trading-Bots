import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _collect_old_files(base: Path, older_than_days: int) -> list[Path]:
    if not base.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    out = []
    for root, _, files in os.walk(base):
        for name in files:
            p = Path(root) / name
            try:
                mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if mt < cutoff:
                out.append(p)
    return out


def _sqlite_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 ** 3)


def _vacuum_sqlite(path: Path) -> bool:
    try:
        conn = sqlite3.connect(str(path))
        conn.execute("VACUUM")
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune old data artifacts by retention policy.")
    parser.add_argument("--decisions-days", type=int, default=45)
    parser.add_argument("--decision-explanations-days", type=int, default=45)
    parser.add_argument("--governance-days", type=int, default=60)
    parser.add_argument("--exports-days", type=int, default=30)
    parser.add_argument("--backup-drills-days", type=int, default=30)
    parser.add_argument("--sqlite-path", default=str(PROJECT_ROOT / "data" / "jsonl_link.sqlite3"))
    parser.add_argument("--sqlite-vacuum-over-gb", type=float, default=6.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    targets = {
        "decisions": (PROJECT_ROOT / "decisions", args.decisions_days),
        "decision_explanations": (PROJECT_ROOT / "decision_explanations", args.decision_explanations_days),
        "governance_watchdog": (PROJECT_ROOT / "governance" / "watchdog", args.governance_days),
        "exports_sql_reports": (PROJECT_ROOT / "exports" / "sql_reports", args.exports_days),
        "backup_drills": (PROJECT_ROOT / "exports" / "backup_drills", args.backup_drills_days),
    }

    to_delete = []
    summary = {}
    for label, (base, days) in targets.items():
        rows = _collect_old_files(base, days)
        to_delete.extend(rows)
        summary[label] = {"candidates": len(rows), "older_than_days": days}

    deleted = 0
    if args.apply:
        for p in to_delete:
            try:
                p.unlink()
                deleted += 1
            except OSError:
                pass

    sqlite_path = Path(args.sqlite_path)
    size_before = _sqlite_size_gb(sqlite_path)
    vacuum_ran = False
    vacuum_ok = None
    if args.apply and sqlite_path.exists() and size_before >= args.sqlite_vacuum_over_gb:
        vacuum_ran = True
        vacuum_ok = _vacuum_sqlite(sqlite_path)
    size_after = _sqlite_size_gb(sqlite_path)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "apply": bool(args.apply),
        "targets": summary,
        "total_candidates": len(to_delete),
        "deleted": deleted,
        "sqlite": {
            "path": str(sqlite_path),
            "size_gb_before": round(size_before, 3),
            "size_gb_after": round(size_after, 3),
            "vacuum_threshold_gb": args.sqlite_vacuum_over_gb,
            "vacuum_ran": vacuum_ran,
            "vacuum_ok": vacuum_ok,
        },
    }

    out = PROJECT_ROOT / "governance" / "health" / "data_retention_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
