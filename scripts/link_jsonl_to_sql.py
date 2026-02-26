import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class FileProgress:
    last_line: int
    mtime: float


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _discover_jsonl_files(project_root: Path, include_globs: List[str], exclude_parts: List[str]) -> List[Path]:
    found: List[Path] = []
    seen = set()

    for pat in include_globs:
        for p in project_root.glob(pat):
            if not p.is_file():
                continue
            s = str(p)
            if any(part and part in s for part in exclude_parts):
                continue
            if s in seen:
                continue
            seen.add(s)
            found.append(p)

    found.sort()
    return found


def _load_state(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not path.exists():
        return {"sqlite": {}, "mysql": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {"sqlite": {}, "mysql": {}}
        obj.setdefault("sqlite", {})
        obj.setdefault("mysql", {})
        return obj
    except Exception:
        return {"sqlite": {}, "mysql": {}}


def _save_state(path: Path, state: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


def _iter_new_lines(path: Path, start_line: int) -> Iterable[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i <= start_line:
                continue
            line = line.strip()
            if not line:
                continue
            yield i, line


def _ensure_sqlite_schema(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            UNIQUE(source_file, line_no)
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_source_rel ON {table}(source_rel)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ingested_at ON {table}(ingested_at)")


def _sqlite_executemany_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    rows: List[Tuple[str, str, int, str, str, str]],
    lock_retries: int,
    lock_retry_delay_seconds: float,
) -> sqlite3.Cursor:
    attempt = 0
    while True:
        try:
            return conn.executemany(sql, rows)
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            is_locked = ("database is locked" in msg) or ("database table is locked" in msg)
            if (not is_locked) or attempt >= max(lock_retries, 0):
                raise
            sleep_s = min(max(lock_retry_delay_seconds, 0.01) * (2 ** attempt), 5.0)
            print(
                f"SQLite busy; retrying batch in {sleep_s:.2f}s "
                f"(attempt {attempt + 1}/{max(lock_retries, 0)})"
            )
            time.sleep(sleep_s)
            attempt += 1


def _sync_file_to_sqlite(
    conn: sqlite3.Connection,
    table: str,
    project_root: Path,
    file_path: Path,
    start_line: int,
    dry_run: bool,
    lock_retries: int,
    lock_retry_delay_seconds: float,
) -> Tuple[int, int, int]:
    inserted = 0
    invalid = 0
    last_line_seen = start_line

    rows: List[Tuple[str, str, int, str, str, str]] = []
    for line_no, raw in _iter_new_lines(file_path, start_line):
        last_line_seen = line_no
        try:
            obj = json.loads(raw)
            payload = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
        except Exception:
            invalid += 1
            continue

        sha1 = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        rows.append(
            (
                str(file_path),
                str(file_path.relative_to(project_root)),
                line_no,
                _now_utc(),
                sha1,
                payload,
            )
        )

        if len(rows) >= 1000:
            if not dry_run:
                cur = _sqlite_executemany_with_retry(
                    conn,
                    f"INSERT OR IGNORE INTO {table} (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                    lock_retries=lock_retries,
                    lock_retry_delay_seconds=lock_retry_delay_seconds,
                )
                inserted += cur.rowcount if cur.rowcount is not None else 0
            else:
                inserted += len(rows)
            rows = []

    if rows:
        if not dry_run:
            cur = _sqlite_executemany_with_retry(
                conn,
                f"INSERT OR IGNORE INTO {table} (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
                lock_retries=lock_retries,
                lock_retry_delay_seconds=lock_retry_delay_seconds,
            )
            inserted += cur.rowcount if cur.rowcount is not None else 0
        else:
            inserted += len(rows)

    return inserted, invalid, last_line_seen


def _mysql_escape(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _mysql_exec(mysql_bin: str, host: str, port: int, user: str, password: str, database: str, sql: str) -> None:
    env = os.environ.copy()
    if password:
        env["MYSQL_PWD"] = password
    cmd = [mysql_bin, "-h", host, "-P", str(port), "-u", user, database, "-e", sql]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "mysql command failed").strip())


def _ensure_mysql_schema(
    mysql_bin: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table: str,
) -> None:
    _mysql_exec(mysql_bin, host, port, user, password, database, f"""
    CREATE TABLE IF NOT EXISTS {table} (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      source_file TEXT NOT NULL,
      source_rel VARCHAR(1024) NOT NULL,
      line_no BIGINT NOT NULL,
      ingested_at VARCHAR(64) NOT NULL,
      payload_sha1 VARCHAR(40) NOT NULL,
      payload_json LONGTEXT NOT NULL,
      UNIQUE KEY uniq_source_line (source_rel(255), line_no),
      KEY idx_ingested_at (ingested_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)


def _sync_file_to_mysql(
    mysql_bin: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table: str,
    project_root: Path,
    file_path: Path,
    start_line: int,
    batch_size: int,
    dry_run: bool,
) -> Tuple[int, int, int]:
    inserted = 0
    invalid = 0
    last_line_seen = start_line
    vals: List[str] = []

    def flush() -> None:
        nonlocal inserted, vals
        if not vals:
            return
        if dry_run:
            inserted += len(vals)
            vals = []
            return
        sql = f"INSERT IGNORE INTO {table} (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json) VALUES " + ",".join(vals) + ";"
        _mysql_exec(mysql_bin, host, port, user, password, database, sql)
        inserted += len(vals)
        vals = []

    for line_no, raw in _iter_new_lines(file_path, start_line):
        last_line_seen = line_no
        try:
            obj = json.loads(raw)
            payload = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
        except Exception:
            invalid += 1
            continue

        source_file = _mysql_escape(str(file_path))
        source_rel = _mysql_escape(str(file_path.relative_to(project_root)))
        ingested_at = _mysql_escape(_now_utc())
        sha1 = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        payload_esc = _mysql_escape(payload)

        vals.append(
            f"('{source_file}','{source_rel}',{line_no},'{ingested_at}','{sha1}','{payload_esc}')"
        )

        if len(vals) >= batch_size:
            flush()

    flush()
    return inserted, invalid, last_line_seen


def main() -> int:
    parser = argparse.ArgumentParser(description="Link all project JSONL files to SQL (SQLite/MySQL).")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--mode", choices=["sqlite", "mysql", "both"], default="both")
    parser.add_argument("--state-file", default=None, help="Path to incremental ingest state JSON.")

    parser.add_argument("--sqlite-db", default=None, help="SQLite database file path.")
    parser.add_argument("--sqlite-table", default="jsonl_records")
    parser.add_argument("--sqlite-timeout-seconds", type=float, default=float(os.getenv("SQLITE_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--sqlite-lock-retries", type=int, default=int(os.getenv("SQLITE_LOCK_RETRIES", "8")))
    parser.add_argument("--sqlite-lock-retry-delay-seconds", type=float, default=float(os.getenv("SQLITE_LOCK_RETRY_DELAY_SECONDS", "0.25")))

    parser.add_argument("--mysql-bin", default=os.getenv("MYSQL_BIN", "/opt/homebrew/bin/mysql"))
    parser.add_argument("--mysql-host", default=os.getenv("MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--mysql-port", type=int, default=int(os.getenv("MYSQL_PORT", "3306")))
    parser.add_argument("--mysql-user", default=os.getenv("MYSQL_USER", "root"))
    parser.add_argument("--mysql-password", default=os.getenv("MYSQL_PASSWORD", ""))
    parser.add_argument("--mysql-database", default=os.getenv("MYSQL_DATABASE", "schwab_trading"))
    parser.add_argument("--mysql-table", default=os.getenv("MYSQL_TABLE", "jsonl_records"))
    parser.add_argument("--mysql-batch-size", type=int, default=int(os.getenv("MYSQL_BATCH_SIZE", "200")))

    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"Project root missing: {project_root}")
        return 2

    include_globs = [
        "paper_trades_paper.jsonl",
        "decision_explanations/**/*.jsonl",
        "decisions/**/*.jsonl",
        "governance/**/*.jsonl",
        "exports/paper_broker_bridge/**/*.jsonl",
        "data/**/*.jsonl",
    ]
    exclude_parts = ["/.git/", "/.venv", "/models/archive/"]

    files = _discover_jsonl_files(project_root, include_globs, exclude_parts)
    if args.max_files > 0:
        files = files[: args.max_files]

    print(f"Discovered JSONL files: {len(files)}")

    state_path = Path(args.state_file).resolve() if args.state_file else (project_root / "governance" / "jsonl_sql_link_state.json")
    state = _load_state(state_path)

    sqlite_conn = None
    if args.mode in {"sqlite", "both"}:
        sqlite_db = Path(args.sqlite_db).resolve() if args.sqlite_db else (project_root / "data" / "jsonl_link.sqlite3")
        if not args.dry_run:
            sqlite_db.parent.mkdir(parents=True, exist_ok=True)
            sqlite_conn = sqlite3.connect(str(sqlite_db), timeout=max(float(args.sqlite_timeout_seconds), 1.0))
            sqlite_conn.execute("PRAGMA journal_mode=WAL")
            sqlite_conn.execute("PRAGMA synchronous=NORMAL")
            sqlite_conn.execute(f"PRAGMA busy_timeout={int(max(float(args.sqlite_timeout_seconds), 1.0) * 1000)}")
            _ensure_sqlite_schema(sqlite_conn, args.sqlite_table)
        print(f"SQLite target: {sqlite_db} table={args.sqlite_table}")

    if args.mode in {"mysql", "both"}:
        if not Path(args.mysql_bin).exists():
            print(f"MySQL CLI not found: {args.mysql_bin}")
            return 2
        if not args.dry_run:
            _ensure_mysql_schema(
                args.mysql_bin,
                args.mysql_host,
                args.mysql_port,
                args.mysql_user,
                args.mysql_password,
                args.mysql_database,
                args.mysql_table,
            )
        print(
            f"MySQL target: host={args.mysql_host}:{args.mysql_port} db={args.mysql_database} table={args.mysql_table} user={args.mysql_user}"
        )

    total_inserted = {"sqlite": 0, "mysql": 0}
    total_invalid = {"sqlite": 0, "mysql": 0}

    try:
        for fp in files:
            rel = str(fp.relative_to(project_root))
            try:
                mtime = fp.stat().st_mtime
            except FileNotFoundError:
                print(f"Skipping vanished file before sync: {rel}")
                continue
            print(f"Syncing: {rel}")

            if args.mode in {"sqlite", "both"}:
                progress = state["sqlite"].get(rel, {"last_line": 0, "mtime": 0.0})
                start_line = int(progress.get("last_line", 0) or 0)
                if mtime < float(progress.get("mtime", 0.0) or 0.0):
                    start_line = 0
                try:
                    ins, bad, last_line = _sync_file_to_sqlite(
                        sqlite_conn,
                        args.sqlite_table,
                        project_root,
                        fp,
                        start_line,
                        args.dry_run,
                        lock_retries=max(args.sqlite_lock_retries, 0),
                        lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                    )
                except FileNotFoundError:
                    print(f"  sqlite skipped vanished file during sync: {rel}")
                    continue
                if not args.dry_run and sqlite_conn is not None:
                    sqlite_conn.commit()
                total_inserted["sqlite"] += ins
                total_invalid["sqlite"] += bad
                state["sqlite"][rel] = {"last_line": float(last_line), "mtime": mtime}
                print(f"  sqlite inserted={ins} invalid={bad} last_line={last_line}")

            if args.mode in {"mysql", "both"}:
                progress = state["mysql"].get(rel, {"last_line": 0, "mtime": 0.0})
                start_line = int(progress.get("last_line", 0) or 0)
                if mtime < float(progress.get("mtime", 0.0) or 0.0):
                    start_line = 0
                try:
                    ins, bad, last_line = _sync_file_to_mysql(
                        args.mysql_bin,
                        args.mysql_host,
                        args.mysql_port,
                        args.mysql_user,
                        args.mysql_password,
                        args.mysql_database,
                        args.mysql_table,
                        project_root,
                        fp,
                        start_line,
                        args.mysql_batch_size,
                        args.dry_run,
                    )
                except FileNotFoundError:
                    print(f"  mysql skipped vanished file during sync: {rel}")
                    continue
                total_inserted["mysql"] += ins
                total_invalid["mysql"] += bad
                state["mysql"][rel] = {"last_line": float(last_line), "mtime": mtime}
                print(f"  mysql inserted={ins} invalid={bad} last_line={last_line}")

        if not args.dry_run:
            _save_state(state_path, state)

        print("Done.")
        if args.mode in {"sqlite", "both"}:
            print(f"SQLite total inserted={total_inserted['sqlite']} invalid={total_invalid['sqlite']}")
        if args.mode in {"mysql", "both"}:
            print(f"MySQL total inserted={total_inserted['mysql']} invalid={total_invalid['mysql']}")
        print(f"State file: {state_path}")
        return 0
    finally:
        if sqlite_conn is not None:
            sqlite_conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
