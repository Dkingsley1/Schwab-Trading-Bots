import argparse
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_INCLUDE_GLOBS = [
    "paper_trades_*.jsonl",
    "live_orders_*.jsonl",
    "exports/trade_logs/**/*.jsonl",
    "decision_explanations/**/*.jsonl",
    "decisions/**/*.jsonl",
    "governance/**/*.jsonl",
    "exports/paper_broker_bridge/**/*.jsonl",
    "data/**/*.jsonl",
]
DEFAULT_EXCLUDE_PARTS = ["/.git/", "/.venv", "/models/archive/"]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_schema_version() -> int:
    try:
        return max(int(os.getenv("LOG_SCHEMA_VERSION", "2")), 1)
    except Exception:
        return 2


def _classify_stream(source_rel: str) -> str:
    rel = str(source_rel or "")
    if rel.startswith("decision_explanations/"):
        return "decision_explanations"
    if rel.startswith("decisions/"):
        return "decisions"
    if rel.startswith("governance/events/"):
        return "governance_events"
    if rel.startswith("governance/watchdog/"):
        return "governance_watchdog"
    if rel.startswith("governance/"):
        return "governance"
    if rel.startswith("exports/trade_logs/"):
        return "trade_logs"
    if rel.startswith("exports/paper_broker_bridge/"):
        return "paper_broker_bridge"
    if rel.startswith("paper_trades_") or rel.startswith("live_orders_"):
        return "top_level_trade_links"
    if rel.startswith("data/"):
        return "data"
    return "other"


def _source_priority(source_rel: str) -> int:
    stream = _classify_stream(source_rel)
    weights = {
        "decision_explanations": 0,
        "decisions": 1,
        "governance_events": 2,
        "governance_watchdog": 3,
        "governance": 4,
        "top_level_trade_links": 5,
        "trade_logs": 6,
        "paper_broker_bridge": 7,
        "data": 8,
        "other": 9,
    }
    return int(weights.get(stream, 9))


def discover_jsonl_files(
    project_root: Path,
    include_globs: Optional[List[str]] = None,
    exclude_parts: Optional[List[str]] = None,
) -> List[Path]:
    include = include_globs or list(DEFAULT_INCLUDE_GLOBS)
    excludes = exclude_parts or list(DEFAULT_EXCLUDE_PARTS)

    found: List[Path] = []
    seen_path = set()
    seen_resolved = set()

    for pat in include:
        for p in project_root.glob(pat):
            if not p.is_file():
                continue
            p_str = str(p)
            if any(part and part in p_str for part in excludes):
                continue
            try:
                resolved = str(p.resolve(strict=False))
            except Exception:
                resolved = p_str
            if p_str in seen_path or resolved in seen_resolved:
                continue
            seen_path.add(p_str)
            seen_resolved.add(resolved)
            found.append(p)

    def _sort_key(path: Path) -> Tuple[int, float, str]:
        try:
            rel = str(path.relative_to(project_root))
        except Exception:
            rel = str(path)
        try:
            mtime = float(path.stat().st_mtime)
        except Exception:
            mtime = 0.0
        return (_source_priority(rel), -mtime, rel)

    found.sort(key=_sort_key)
    return found


def _discover_jsonl_files(project_root: Path, include_globs: List[str], exclude_parts: List[str]) -> List[Path]:
    # Backward-compatible helper kept for tests and internal callers.
    return discover_jsonl_files(project_root, include_globs=include_globs, exclude_parts=exclude_parts)


def _load_state(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
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


def _save_state(path: Path, state: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    except Exception:
        return


def _parse_ts_utc(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_event_ts_utc(obj: Dict[str, Any]) -> Optional[datetime]:
    md = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    keys = [
        "timestamp_utc",
        "ts_utc",
        "event_timestamp_utc",
        "created_at",
        "timestamp",
    ]
    for key in keys:
        dt = _parse_ts_utc(obj.get(key))
        if dt is not None:
            return dt
    for key in keys:
        dt = _parse_ts_utc(md.get(key))
        if dt is not None:
            return dt
    return None


def _iter_new_lines(path: Path, start_line: int, start_offset_bytes: int = 0) -> Iterable[Tuple[int, str, int]]:
    line_no = max(int(start_line), 0)
    offset = max(int(start_offset_bytes), 0)

    with open(path, "rb") as f:
        if offset > 0:
            try:
                f.seek(offset)
            except Exception:
                f.seek(0)
                line_no = 0

        while True:
            raw = f.readline()
            if not raw:
                break
            line_no += 1
            out_offset = int(f.tell())
            line = raw.rstrip(b"\r\n")
            if not line.strip():
                continue
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                text = line.decode("utf-8", errors="replace")
            yield line_no, text, out_offset


def _extract_correlation_fields(obj: Dict[str, Any]) -> Tuple[str, str, str, str, int]:
    md = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

    def _pick(key: str) -> str:
        val = obj.get(key)
        if val is None or val == "":
            val = md.get(key)
        return str(val) if (val is not None and val != "") else ""

    run_id = _pick("run_id")
    iter_id = _pick("iter_id")
    decision_id = _pick("decision_id")
    parent_decision_id = _pick("parent_decision_id")

    raw_schema = obj.get("log_schema_version")
    if raw_schema is None:
        raw_schema = md.get("log_schema_version")
    try:
        schema_version = max(int(raw_schema), 0)
    except Exception:
        schema_version = 0
    if schema_version <= 0:
        schema_version = _log_schema_version()

    return run_id, iter_id, decision_id, parent_decision_id, schema_version


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _derive_start_cursor(progress: Dict[str, Any], stat: os.stat_result) -> Tuple[int, int, str]:
    start_line = int(float(progress.get("last_line", 0) or 0))
    start_offset = int(float(progress.get("last_offset_bytes", 0) or 0))
    prev_mtime = float(progress.get("mtime", 0.0) or 0.0)

    prev_inode = int(float(progress.get("file_inode", 0) or 0))
    prev_size = int(float(progress.get("file_size_bytes", 0) or 0))

    if prev_inode > 0 and int(stat.st_ino) != prev_inode:
        return 0, 0, "inode_changed"
    if prev_size > 0 and int(stat.st_size) < prev_size:
        return 0, 0, "size_shrank"
    if float(stat.st_mtime) < prev_mtime:
        return 0, 0, "mtime_rewound"
    if start_offset > int(stat.st_size):
        return 0, 0, "offset_past_eof"

    return max(start_line, 0), max(start_offset, 0), ""


def _record_top_pending(
    rows: List[Dict[str, Any]],
    *,
    source_rel: str,
    pending_lines: int,
    oldest_age_seconds: float,
    total_lines: int,
    last_line: int,
    top_n: int,
) -> None:
    if pending_lines <= 0:
        return
    rows.append(
        {
            "source_rel": str(source_rel),
            "stream": _classify_stream(source_rel),
            "pending_lines": int(pending_lines),
            "oldest_pending_age_seconds": round(float(oldest_age_seconds), 3),
            "total_lines": int(total_lines),
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


class LatencyAccumulator:
    def __init__(self, reservoir_size: int = 2048) -> None:
        self.reservoir_size = max(int(reservoir_size), 128)
        self.count = 0
        self.total_seconds = 0.0
        self.max_seconds = 0.0
        self.slo_breaches_300s = 0
        self._samples: List[float] = []

    def add(self, latency_seconds: float) -> None:
        val = max(float(latency_seconds), 0.0)
        self.count += 1
        self.total_seconds += val
        if val > self.max_seconds:
            self.max_seconds = val
        if val > 300.0:
            self.slo_breaches_300s += 1

        if len(self._samples) < self.reservoir_size:
            self._samples.append(val)
            return

        idx = random.randint(0, self.count - 1)
        if idx < self.reservoir_size:
            self._samples[idx] = val

    def snapshot(self) -> Dict[str, Any]:
        if self.count <= 0:
            return {
                "count": 0,
                "p50_seconds": 0.0,
                "p95_seconds": 0.0,
                "max_seconds": 0.0,
                "mean_seconds": 0.0,
                "slo_breach_ratio_gt_300s": 0.0,
            }

        vals = sorted(self._samples)

        def _pct(p: float) -> float:
            if not vals:
                return 0.0
            i = min(max(int(round((len(vals) - 1) * p)), 0), len(vals) - 1)
            return float(vals[i])

        return {
            "count": int(self.count),
            "p50_seconds": round(_pct(0.50), 3),
            "p95_seconds": round(_pct(0.95), 3),
            "max_seconds": round(float(self.max_seconds), 3),
            "mean_seconds": round(float(self.total_seconds) / max(int(self.count), 1), 3),
            "slo_breach_ratio_gt_300s": round(float(self.slo_breaches_300s) / max(int(self.count), 1), 6),
        }


def _latency_payload(acc_all: LatencyAccumulator, by_stream: Dict[str, LatencyAccumulator]) -> Dict[str, Any]:
    stream_rows = {}
    for stream, acc in sorted(by_stream.items()):
        snap = acc.snapshot()
        if int(snap.get("count", 0)) > 0:
            stream_rows[str(stream)] = snap
    return {
        "all": acc_all.snapshot(),
        "by_stream": stream_rows,
    }


def _ensure_latency_bucket(
    store: Dict[str, Dict[str, Any]],
    mode: str,
    stream: str,
) -> Tuple[LatencyAccumulator, LatencyAccumulator]:
    mode_obj = store.setdefault(mode, {"all": LatencyAccumulator(), "by_stream": {}})
    by_stream = mode_obj.setdefault("by_stream", {})
    stream_acc = by_stream.get(stream)
    if stream_acc is None:
        stream_acc = LatencyAccumulator()
        by_stream[stream] = stream_acc
    return mode_obj["all"], stream_acc


def _log_invalid_line(
    *,
    invalid_log_path: Optional[Path],
    mode: str,
    source_rel: str,
    line_no: int,
    raw: str,
    error: Exception,
    run_id: str,
    iter_id: str,
) -> None:
    if invalid_log_path is None:
        return
    _append_jsonl(
        invalid_log_path,
        {
            "timestamp_utc": _now_utc(),
            "event": "ingest_invalid_json",
            "mode": str(mode),
            "source_rel": str(source_rel),
            "stream": _classify_stream(source_rel),
            "line_no": int(line_no),
            "error": str(error),
            "raw_sample": str(raw)[:512],
            "run_id": str(run_id or ""),
            "iter_id": str(iter_id or ""),
            "log_schema_version": _log_schema_version(),
        },
    )


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
            run_id TEXT,
            iter_id TEXT,
            decision_id TEXT,
            parent_decision_id TEXT,
            log_schema_version INTEGER,
            UNIQUE(source_file, line_no)
        )
        """
    )

    try:
        cols = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    except Exception:
        cols = set()

    expected_cols = {
        "run_id": "TEXT",
        "iter_id": "TEXT",
        "decision_id": "TEXT",
        "parent_decision_id": "TEXT",
        "log_schema_version": "INTEGER",
    }
    for col, col_type in expected_cols.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_source_rel ON {table}(source_rel)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ingested_at ON {table}(ingested_at)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_run_id ON {table}(run_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_iter_id ON {table}(iter_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_decision_id ON {table}(decision_id)")


def _sqlite_executemany_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    rows: List[Tuple[Any, ...]],
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
    conn: Optional[sqlite3.Connection],
    table: str,
    project_root: Path,
    file_path: Path,
    start_line: int,
    start_offset_bytes: int,
    dry_run: bool,
    lock_retries: int,
    lock_retry_delay_seconds: float,
    latency_all: Optional[LatencyAccumulator],
    latency_stream: Optional[LatencyAccumulator],
    invalid_log_path: Optional[Path],
    invalid_sample_limit: int,
    run_id: str,
    iter_id: str,
) -> Dict[str, Any]:
    inserted = 0
    invalid = 0
    invalid_logged = 0
    last_line_seen = max(int(start_line), 0)
    last_offset_seen = max(int(start_offset_bytes), 0)
    source_rel = str(file_path.relative_to(project_root))

    rows: List[Tuple[Any, ...]] = []
    for line_no, raw, next_offset in _iter_new_lines(file_path, start_line, start_offset_bytes):
        last_line_seen = int(line_no)
        last_offset_seen = int(next_offset)
        try:
            obj = json.loads(raw)
            payload = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
        except Exception as exc:
            invalid += 1
            if invalid_logged < max(int(invalid_sample_limit), 0):
                _log_invalid_line(
                    invalid_log_path=invalid_log_path,
                    mode="sqlite",
                    source_rel=source_rel,
                    line_no=line_no,
                    raw=raw,
                    error=exc,
                    run_id=run_id,
                    iter_id=iter_id,
                )
                invalid_logged += 1
            continue

        event_ts = _extract_event_ts_utc(obj)
        if event_ts is not None:
            latency_s = max(time.time() - event_ts.timestamp(), 0.0)
            if latency_all is not None:
                latency_all.add(latency_s)
            if latency_stream is not None:
                latency_stream.add(latency_s)

        run_id_row, iter_id_row, decision_id, parent_decision_id, schema_version = _extract_correlation_fields(obj)
        sha1 = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        rows.append(
            (
                str(file_path),
                source_rel,
                line_no,
                _now_utc(),
                sha1,
                payload,
                run_id_row,
                iter_id_row,
                decision_id,
                parent_decision_id,
                schema_version,
            )
        )

        if len(rows) >= 1000:
            if not dry_run:
                if conn is None:
                    raise RuntimeError("sqlite connection missing")
                cur = _sqlite_executemany_with_retry(
                    conn,
                    f"INSERT OR IGNORE INTO {table} (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json, run_id, iter_id, decision_id, parent_decision_id, log_schema_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            if conn is None:
                raise RuntimeError("sqlite connection missing")
            cur = _sqlite_executemany_with_retry(
                conn,
                f"INSERT OR IGNORE INTO {table} (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json, run_id, iter_id, decision_id, parent_decision_id, log_schema_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
                lock_retries=lock_retries,
                lock_retry_delay_seconds=lock_retry_delay_seconds,
            )
            inserted += cur.rowcount if cur.rowcount is not None else 0
        else:
            inserted += len(rows)

    return {
        "inserted": int(inserted),
        "invalid": int(invalid),
        "invalid_samples_logged": int(invalid_logged),
        "last_line": int(last_line_seen),
        "last_offset_bytes": int(last_offset_seen),
    }


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


def _mysql_exec_allow_duplicate(
    mysql_bin: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    sql: str,
    *,
    duplicate_markers: Tuple[str, ...],
) -> None:
    try:
        _mysql_exec(mysql_bin, host, port, user, password, database, sql)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if any(marker in msg for marker in duplicate_markers):
            return
        raise


def _ensure_mysql_schema(
    mysql_bin: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table: str,
) -> None:
    _mysql_exec(
        mysql_bin,
        host,
        port,
        user,
        password,
        database,
        f"""
    CREATE TABLE IF NOT EXISTS {table} (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      source_file TEXT NOT NULL,
      source_rel VARCHAR(1024) NOT NULL,
      line_no BIGINT NOT NULL,
      ingested_at VARCHAR(64) NOT NULL,
      payload_sha1 VARCHAR(40) NOT NULL,
      payload_json LONGTEXT NOT NULL,
      run_id VARCHAR(192) NULL,
      iter_id VARCHAR(192) NULL,
      decision_id VARCHAR(192) NULL,
      parent_decision_id VARCHAR(192) NULL,
      log_schema_version INT NULL,
      UNIQUE KEY uniq_source_line (source_rel(255), line_no),
      KEY idx_ingested_at (ingested_at),
      KEY idx_run_id (run_id),
      KEY idx_iter_id (iter_id),
      KEY idx_decision_id (decision_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    )

    for sql in [
        f"ALTER TABLE {table} ADD COLUMN run_id VARCHAR(192) NULL",
        f"ALTER TABLE {table} ADD COLUMN iter_id VARCHAR(192) NULL",
        f"ALTER TABLE {table} ADD COLUMN decision_id VARCHAR(192) NULL",
        f"ALTER TABLE {table} ADD COLUMN parent_decision_id VARCHAR(192) NULL",
        f"ALTER TABLE {table} ADD COLUMN log_schema_version INT NULL",
    ]:
        _mysql_exec_allow_duplicate(
            mysql_bin,
            host,
            port,
            user,
            password,
            database,
            sql,
            duplicate_markers=("duplicate column name", "error 1060"),
        )

    for sql in [
        f"ALTER TABLE {table} ADD INDEX idx_run_id (run_id)",
        f"ALTER TABLE {table} ADD INDEX idx_iter_id (iter_id)",
        f"ALTER TABLE {table} ADD INDEX idx_decision_id (decision_id)",
    ]:
        _mysql_exec_allow_duplicate(
            mysql_bin,
            host,
            port,
            user,
            password,
            database,
            sql,
            duplicate_markers=("duplicate key name", "error 1061"),
        )


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
    start_offset_bytes: int,
    batch_size: int,
    dry_run: bool,
    latency_all: Optional[LatencyAccumulator],
    latency_stream: Optional[LatencyAccumulator],
    invalid_log_path: Optional[Path],
    invalid_sample_limit: int,
    run_id: str,
    iter_id: str,
) -> Dict[str, Any]:
    inserted = 0
    invalid = 0
    invalid_logged = 0
    last_line_seen = max(int(start_line), 0)
    last_offset_seen = max(int(start_offset_bytes), 0)
    vals: List[str] = []
    source_rel = str(file_path.relative_to(project_root))

    def flush() -> None:
        nonlocal inserted, vals
        if not vals:
            return
        if dry_run:
            inserted += len(vals)
            vals = []
            return
        sql = (
            f"INSERT IGNORE INTO {table} "
            "(source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json, run_id, iter_id, decision_id, parent_decision_id, log_schema_version) VALUES "
            + ",".join(vals)
            + ";"
        )
        _mysql_exec(mysql_bin, host, port, user, password, database, sql)
        inserted += len(vals)
        vals = []

    for line_no, raw, next_offset in _iter_new_lines(file_path, start_line, start_offset_bytes):
        last_line_seen = int(line_no)
        last_offset_seen = int(next_offset)
        try:
            obj = json.loads(raw)
            payload = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
        except Exception as exc:
            invalid += 1
            if invalid_logged < max(int(invalid_sample_limit), 0):
                _log_invalid_line(
                    invalid_log_path=invalid_log_path,
                    mode="mysql",
                    source_rel=source_rel,
                    line_no=line_no,
                    raw=raw,
                    error=exc,
                    run_id=run_id,
                    iter_id=iter_id,
                )
                invalid_logged += 1
            continue

        event_ts = _extract_event_ts_utc(obj)
        if event_ts is not None:
            latency_s = max(time.time() - event_ts.timestamp(), 0.0)
            if latency_all is not None:
                latency_all.add(latency_s)
            if latency_stream is not None:
                latency_stream.add(latency_s)

        run_id_row, iter_id_row, decision_id, parent_decision_id, schema_version = _extract_correlation_fields(obj)

        source_file = _mysql_escape(str(file_path))
        source_rel_esc = _mysql_escape(source_rel)
        ingested_at = _mysql_escape(_now_utc())
        sha1 = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        payload_esc = _mysql_escape(payload)

        vals.append(
            "("
            f"'{source_file}','{source_rel_esc}',{line_no},'{ingested_at}','{sha1}','{payload_esc}',"
            f"'{_mysql_escape(run_id_row)}','{_mysql_escape(iter_id_row)}','{_mysql_escape(decision_id)}','{_mysql_escape(parent_decision_id)}',"
            f"{int(schema_version)}"
            ")"
        )

        if len(vals) >= batch_size:
            flush()

    flush()

    return {
        "inserted": int(inserted),
        "invalid": int(invalid),
        "invalid_samples_logged": int(invalid_logged),
        "last_line": int(last_line_seen),
        "last_offset_bytes": int(last_offset_seen),
    }


def _journal_event(paths: List[Path], payload: Dict[str, Any]) -> None:
    for path in paths:
        _append_jsonl(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Link all project JSONL files to SQL (SQLite/MySQL).")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--mode", choices=["sqlite", "mysql", "both"], default="both")
    parser.add_argument("--state-file", default=None, help="Path to incremental ingest state JSON.")

    parser.add_argument("--sqlite-db", default=None, help="SQLite database file path.")
    parser.add_argument("--sqlite-table", default="jsonl_records")
    parser.add_argument("--sqlite-timeout-seconds", type=float, default=float(os.getenv("SQLITE_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--sqlite-lock-retries", type=int, default=int(os.getenv("SQLITE_LOCK_RETRIES", "8")))
    parser.add_argument(
        "--sqlite-lock-retry-delay-seconds",
        type=float,
        default=float(os.getenv("SQLITE_LOCK_RETRY_DELAY_SECONDS", "0.25")),
    )

    parser.add_argument("--mysql-bin", default=os.getenv("MYSQL_BIN", "/opt/homebrew/bin/mysql"))
    parser.add_argument("--mysql-host", default=os.getenv("MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--mysql-port", type=int, default=int(os.getenv("MYSQL_PORT", "3306")))
    parser.add_argument("--mysql-user", default=os.getenv("MYSQL_USER", "root"))
    parser.add_argument("--mysql-password", default=os.getenv("MYSQL_PASSWORD", ""))
    parser.add_argument("--mysql-database", default=os.getenv("MYSQL_DATABASE", "schwab_trading"))
    parser.add_argument("--mysql-table", default=os.getenv("MYSQL_TABLE", "jsonl_records"))
    parser.add_argument("--mysql-batch-size", type=int, default=int(os.getenv("MYSQL_BATCH_SIZE", "200")))

    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--top-pending-files", type=int, default=int(os.getenv("INGEST_TOP_PENDING_FILES", "10")))
    parser.add_argument("--invalid-sample-limit", type=int, default=int(os.getenv("INGEST_INVALID_SAMPLE_LIMIT", "25")))
    parser.add_argument("--invalid-log-file", default=os.getenv("INGEST_INVALID_LOG_FILE", ""))
    parser.add_argument("--journal-file", default=os.getenv("INGEST_JOURNAL_FILE", ""))
    parser.add_argument("--journal-events-file", default=os.getenv("INGEST_JOURNAL_EVENTS_FILE", ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"Project root missing: {project_root}")
        return 2

    files = discover_jsonl_files(project_root)
    if args.max_files > 0:
        files = files[: args.max_files]

    print(f"Discovered JSONL files: {len(files)}")

    state_path = (
        Path(args.state_file).resolve()
        if args.state_file
        else (project_root / "governance" / "jsonl_sql_link_state.json")
    )
    state = _load_state(state_path)

    day_utc = datetime.now(timezone.utc).strftime("%Y%m%d")
    default_invalid_log = project_root / "governance" / "events" / f"jsonl_ingestion_invalid_{day_utc}.jsonl"
    invalid_log_path = Path(args.invalid_log_file).resolve() if args.invalid_log_file else default_invalid_log

    default_journal_latest = project_root / "governance" / "health" / "jsonl_ingest_batch_journal_latest.jsonl"
    default_journal_daily = project_root / "governance" / "events" / f"jsonl_ingest_batches_{day_utc}.jsonl"
    journal_paths = [
        Path(args.journal_file).resolve() if args.journal_file else default_journal_latest,
        Path(args.journal_events_file).resolve() if args.journal_events_file else default_journal_daily,
    ]

    run_id = str(os.getenv("CORRELATION_RUN_ID", "") or "").strip()
    iter_id = str(os.getenv("CORRELATION_ITER_ID", "") or "").strip()
    ingest_run_id = run_id or f"ingest-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"

    sqlite_conn: Optional[sqlite3.Connection] = None
    if args.mode in {"sqlite", "both"}:
        sqlite_db = Path(args.sqlite_db).resolve() if args.sqlite_db else (project_root / "data" / "jsonl_link.sqlite3")
        if not args.dry_run:
            sqlite_db.parent.mkdir(parents=True, exist_ok=True)
            sqlite_conn = sqlite3.connect(
                str(sqlite_db),
                timeout=max(float(args.sqlite_timeout_seconds), 1.0),
            )
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
    total_invalid_samples = {"sqlite": 0, "mysql": 0}
    lag_metrics = {
        "sqlite": {
            "pending_lines": 0,
            "oldest_uningested_age_seconds": 0.0,
            "files_with_pending": 0,
            "top_pending_files": [],
        },
        "mysql": {
            "pending_lines": 0,
            "oldest_uningested_age_seconds": 0.0,
            "files_with_pending": 0,
            "top_pending_files": [],
        },
    }
    latency_metrics: Dict[str, Dict[str, Any]] = {
        "sqlite": {"all": LatencyAccumulator(), "by_stream": {}},
        "mysql": {"all": LatencyAccumulator(), "by_stream": {}},
    }

    try:
        for fp in files:
            rel = str(fp.relative_to(project_root))
            stream = _classify_stream(rel)
            try:
                st = fp.stat()
                mtime = float(st.st_mtime)
            except FileNotFoundError:
                print(f"Skipping vanished file before sync: {rel}")
                continue

            print(f"Syncing: {rel}")
            total_lines = _count_lines(fp)

            if args.mode in {"sqlite", "both"}:
                progress = state["sqlite"].get(rel, {"last_line": 0, "mtime": 0.0})
                start_line, start_offset, reset_reason = _derive_start_cursor(progress, st)
                start_evt = {
                    "timestamp_utc": _now_utc(),
                    "event": "file_start",
                    "ingest_run_id": ingest_run_id,
                    "mode": "sqlite",
                    "source_rel": rel,
                    "stream": stream,
                    "start_line": int(start_line),
                    "start_offset_bytes": int(start_offset),
                    "reset_reason": str(reset_reason),
                }
                _journal_event(journal_paths, start_evt)

                lat_all, lat_stream = _ensure_latency_bucket(latency_metrics, "sqlite", stream)
                started_ts = time.time()
                try:
                    result = _sync_file_to_sqlite(
                        sqlite_conn,
                        args.sqlite_table,
                        project_root,
                        fp,
                        start_line,
                        start_offset,
                        args.dry_run,
                        lock_retries=max(args.sqlite_lock_retries, 0),
                        lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                        latency_all=lat_all,
                        latency_stream=lat_stream,
                        invalid_log_path=invalid_log_path,
                        invalid_sample_limit=max(args.invalid_sample_limit, 0),
                        run_id=run_id,
                        iter_id=iter_id,
                    )
                except FileNotFoundError:
                    _journal_event(
                        journal_paths,
                        {
                            "timestamp_utc": _now_utc(),
                            "event": "file_failed",
                            "ingest_run_id": ingest_run_id,
                            "mode": "sqlite",
                            "source_rel": rel,
                            "stream": stream,
                            "error": "file_vanished_during_sync",
                        },
                    )
                    print(f"  sqlite skipped vanished file during sync: {rel}")
                    continue
                except Exception as exc:
                    _journal_event(
                        journal_paths,
                        {
                            "timestamp_utc": _now_utc(),
                            "event": "file_failed",
                            "ingest_run_id": ingest_run_id,
                            "mode": "sqlite",
                            "source_rel": rel,
                            "stream": stream,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise

                if not args.dry_run and sqlite_conn is not None:
                    sqlite_conn.commit()

                total_inserted["sqlite"] += int(result["inserted"])
                total_invalid["sqlite"] += int(result["invalid"])
                total_invalid_samples["sqlite"] += int(result["invalid_samples_logged"])

                try:
                    post_st = fp.stat()
                except Exception:
                    post_st = st

                state["sqlite"][rel] = {
                    "last_line": int(result["last_line"]),
                    "last_offset_bytes": int(result["last_offset_bytes"]),
                    "mtime": float(post_st.st_mtime),
                    "file_inode": int(post_st.st_ino),
                    "file_size_bytes": int(post_st.st_size),
                }

                pending_lines = max(int(total_lines) - int(result["last_line"]), 0)
                oldest_age = max(time.time() - mtime, 0.0) if pending_lines > 0 else 0.0
                lag_metrics["sqlite"]["pending_lines"] += pending_lines
                lag_metrics["sqlite"]["oldest_uningested_age_seconds"] = max(
                    float(lag_metrics["sqlite"].get("oldest_uningested_age_seconds", 0.0) or 0.0),
                    float(oldest_age),
                )
                if pending_lines > 0:
                    lag_metrics["sqlite"]["files_with_pending"] += 1
                    _record_top_pending(
                        lag_metrics["sqlite"]["top_pending_files"],
                        source_rel=rel,
                        pending_lines=pending_lines,
                        oldest_age_seconds=oldest_age,
                        total_lines=total_lines,
                        last_line=int(result["last_line"]),
                        top_n=max(int(args.top_pending_files), 1),
                    )

                _journal_event(
                    journal_paths,
                    {
                        "timestamp_utc": _now_utc(),
                        "event": "file_complete",
                        "ingest_run_id": ingest_run_id,
                        "mode": "sqlite",
                        "source_rel": rel,
                        "stream": stream,
                        "inserted": int(result["inserted"]),
                        "invalid": int(result["invalid"]),
                        "invalid_samples_logged": int(result["invalid_samples_logged"]),
                        "last_line": int(result["last_line"]),
                        "last_offset_bytes": int(result["last_offset_bytes"]),
                        "pending_lines": int(pending_lines),
                        "duration_seconds": round(max(time.time() - started_ts, 0.0), 4),
                    },
                )

                print(
                    f"  sqlite inserted={result['inserted']} invalid={result['invalid']} "
                    f"last_line={result['last_line']} last_offset={result['last_offset_bytes']} "
                    f"pending_lines={pending_lines}"
                )

            if args.mode in {"mysql", "both"}:
                progress = state["mysql"].get(rel, {"last_line": 0, "mtime": 0.0})
                start_line, start_offset, reset_reason = _derive_start_cursor(progress, st)
                _journal_event(
                    journal_paths,
                    {
                        "timestamp_utc": _now_utc(),
                        "event": "file_start",
                        "ingest_run_id": ingest_run_id,
                        "mode": "mysql",
                        "source_rel": rel,
                        "stream": stream,
                        "start_line": int(start_line),
                        "start_offset_bytes": int(start_offset),
                        "reset_reason": str(reset_reason),
                    },
                )

                lat_all, lat_stream = _ensure_latency_bucket(latency_metrics, "mysql", stream)
                started_ts = time.time()
                try:
                    result = _sync_file_to_mysql(
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
                        start_offset,
                        args.mysql_batch_size,
                        args.dry_run,
                        latency_all=lat_all,
                        latency_stream=lat_stream,
                        invalid_log_path=invalid_log_path,
                        invalid_sample_limit=max(args.invalid_sample_limit, 0),
                        run_id=run_id,
                        iter_id=iter_id,
                    )
                except FileNotFoundError:
                    _journal_event(
                        journal_paths,
                        {
                            "timestamp_utc": _now_utc(),
                            "event": "file_failed",
                            "ingest_run_id": ingest_run_id,
                            "mode": "mysql",
                            "source_rel": rel,
                            "stream": stream,
                            "error": "file_vanished_during_sync",
                        },
                    )
                    print(f"  mysql skipped vanished file during sync: {rel}")
                    continue
                except Exception as exc:
                    _journal_event(
                        journal_paths,
                        {
                            "timestamp_utc": _now_utc(),
                            "event": "file_failed",
                            "ingest_run_id": ingest_run_id,
                            "mode": "mysql",
                            "source_rel": rel,
                            "stream": stream,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise

                total_inserted["mysql"] += int(result["inserted"])
                total_invalid["mysql"] += int(result["invalid"])
                total_invalid_samples["mysql"] += int(result["invalid_samples_logged"])

                try:
                    post_st = fp.stat()
                except Exception:
                    post_st = st

                state["mysql"][rel] = {
                    "last_line": int(result["last_line"]),
                    "last_offset_bytes": int(result["last_offset_bytes"]),
                    "mtime": float(post_st.st_mtime),
                    "file_inode": int(post_st.st_ino),
                    "file_size_bytes": int(post_st.st_size),
                }

                pending_lines = max(int(total_lines) - int(result["last_line"]), 0)
                oldest_age = max(time.time() - mtime, 0.0) if pending_lines > 0 else 0.0
                lag_metrics["mysql"]["pending_lines"] += pending_lines
                lag_metrics["mysql"]["oldest_uningested_age_seconds"] = max(
                    float(lag_metrics["mysql"].get("oldest_uningested_age_seconds", 0.0) or 0.0),
                    float(oldest_age),
                )
                if pending_lines > 0:
                    lag_metrics["mysql"]["files_with_pending"] += 1
                    _record_top_pending(
                        lag_metrics["mysql"]["top_pending_files"],
                        source_rel=rel,
                        pending_lines=pending_lines,
                        oldest_age_seconds=oldest_age,
                        total_lines=total_lines,
                        last_line=int(result["last_line"]),
                        top_n=max(int(args.top_pending_files), 1),
                    )

                _journal_event(
                    journal_paths,
                    {
                        "timestamp_utc": _now_utc(),
                        "event": "file_complete",
                        "ingest_run_id": ingest_run_id,
                        "mode": "mysql",
                        "source_rel": rel,
                        "stream": stream,
                        "inserted": int(result["inserted"]),
                        "invalid": int(result["invalid"]),
                        "invalid_samples_logged": int(result["invalid_samples_logged"]),
                        "last_line": int(result["last_line"]),
                        "last_offset_bytes": int(result["last_offset_bytes"]),
                        "pending_lines": int(pending_lines),
                        "duration_seconds": round(max(time.time() - started_ts, 0.0), 4),
                    },
                )

                print(
                    f"  mysql inserted={result['inserted']} invalid={result['invalid']} "
                    f"last_line={result['last_line']} last_offset={result['last_offset_bytes']} "
                    f"pending_lines={pending_lines}"
                )

        if not args.dry_run:
            _save_state(state_path, state)

        health_payload = {
            "timestamp_utc": _now_utc(),
            "log_schema_version": _log_schema_version(),
            "run_id": run_id,
            "iter_id": iter_id,
            "ingest_run_id": ingest_run_id,
            "mode": args.mode,
            "project_root": str(project_root),
            "state_file": str(state_path),
            "checkpoint_mode": "line_offset_inode_v2",
            "files_discovered": int(len(files)),
            "invalid_log_file": str(invalid_log_path),
            "journal_files": [str(p) for p in journal_paths],
            "sqlite": {
                "inserted": int(total_inserted["sqlite"]),
                "invalid": int(total_invalid["sqlite"]),
                "invalid_samples_logged": int(total_invalid_samples["sqlite"]),
                "pending_lines": int(lag_metrics["sqlite"]["pending_lines"]),
                "oldest_uningested_age_seconds": float(lag_metrics["sqlite"]["oldest_uningested_age_seconds"]),
                "files_with_pending": int(lag_metrics["sqlite"]["files_with_pending"]),
                "top_pending_files": list(lag_metrics["sqlite"]["top_pending_files"]),
            },
            "mysql": {
                "inserted": int(total_inserted["mysql"]),
                "invalid": int(total_invalid["mysql"]),
                "invalid_samples_logged": int(total_invalid_samples["mysql"]),
                "pending_lines": int(lag_metrics["mysql"]["pending_lines"]),
                "oldest_uningested_age_seconds": float(lag_metrics["mysql"]["oldest_uningested_age_seconds"]),
                "files_with_pending": int(lag_metrics["mysql"]["files_with_pending"]),
                "top_pending_files": list(lag_metrics["mysql"]["top_pending_files"]),
            },
            "latency_slo": {
                "sqlite": _latency_payload(
                    latency_metrics["sqlite"]["all"],
                    latency_metrics["sqlite"]["by_stream"],
                ),
                "mysql": _latency_payload(
                    latency_metrics["mysql"]["all"],
                    latency_metrics["mysql"]["by_stream"],
                ),
            },
        }

        health_dir = project_root / "governance" / "health"
        health_dir.mkdir(parents=True, exist_ok=True)
        health_latest = health_dir / "jsonl_sql_ingestion_health_latest.json"
        with open(health_latest, "w", encoding="utf-8") as f:
            json.dump(health_payload, f, ensure_ascii=True, indent=2)

        print("Done.")
        if args.mode in {"sqlite", "both"}:
            sqlite_lat = health_payload["latency_slo"]["sqlite"]["all"]
            print(
                f"SQLite total inserted={total_inserted['sqlite']} invalid={total_invalid['sqlite']} "
                f"pending={lag_metrics['sqlite']['pending_lines']} "
                f"oldest_pending_age_s={lag_metrics['sqlite']['oldest_uningested_age_seconds']:.1f} "
                f"p95_latency_s={float(sqlite_lat.get('p95_seconds', 0.0) or 0.0):.1f}"
            )
        if args.mode in {"mysql", "both"}:
            mysql_lat = health_payload["latency_slo"]["mysql"]["all"]
            print(
                f"MySQL total inserted={total_inserted['mysql']} invalid={total_invalid['mysql']} "
                f"pending={lag_metrics['mysql']['pending_lines']} "
                f"oldest_pending_age_s={lag_metrics['mysql']['oldest_uningested_age_seconds']:.1f} "
                f"p95_latency_s={float(mysql_lat.get('p95_seconds', 0.0) or 0.0):.1f}"
            )
        print(f"State file: {state_path}")
        print(f"Health summary: {health_latest}")
        return 0
    finally:
        if sqlite_conn is not None:
            sqlite_conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
