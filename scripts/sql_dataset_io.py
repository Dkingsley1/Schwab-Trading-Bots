import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import orjson as _fast_json
except Exception:
    _fast_json = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"


def _connect_readonly(sqlite_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(sqlite_path), timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _json_loads(raw: Any) -> Any:
    if _fast_json is not None:
        try:
            if isinstance(raw, str):
                return _fast_json.loads(raw)
            return _fast_json.loads(raw or b"")
        except Exception:
            pass
    return json.loads(raw)


def resolve_sqlite_path(raw: Any = None) -> Path:
    text = str(raw or "").strip()
    if text:
        return Path(text).expanduser().resolve()
    return DEFAULT_SQLITE_PATH.resolve()


def source_rel_for_path(project_root: Path, path: Path) -> str:
    path_obj = Path(path).expanduser()
    root_obj = Path(project_root).expanduser()
    candidates = [
        (path_obj, root_obj),
        (path_obj.absolute(), root_obj.absolute()),
        (path_obj.resolve(strict=False), root_obj.resolve(strict=False)),
    ]
    for child, parent in candidates:
        try:
            return str(child.relative_to(parent))
        except Exception:
            continue
    raise ValueError(f"path {path} is not under project root {project_root}")


def source_rels_for_paths(project_root: Path, paths: Sequence[Path]) -> List[str]:
    out: List[str] = []
    for path in paths:
        try:
            out.append(source_rel_for_path(project_root, path))
        except Exception:
            continue
    return out


def _chunked(items: Sequence[str], size: int = 500) -> Iterator[List[str]]:
    chunk_size = max(int(size), 1)
    for i in range(0, len(items), chunk_size):
        yield list(items[i : i + chunk_size])


def _literal_like_prefix(pattern: str) -> str:
    out: List[str] = []
    escaped = False
    for char in str(pattern or ""):
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"%", "_"}:
            break
        out.append(char)
    return "".join(out)


def _prefix_upper_bound(prefix: str) -> str:
    if not prefix:
        return ""
    chars = list(prefix)
    chars[-1] = chr(ord(chars[-1]) + 1)
    return "".join(chars)


def source_rels_present_in_sqlite(
    *,
    sqlite_path: Path,
    source_rels: Sequence[str],
    table: str = "jsonl_records",
) -> Set[str]:
    if (not source_rels) or (not sqlite_path.exists()):
        return set()

    present: Set[str] = set()
    try:
        conn = _connect_readonly(sqlite_path)
    except sqlite3.Error:
        return present
    try:
        for chunk in _chunked(list(source_rels), size=100):
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT DISTINCT source_rel FROM {table} WHERE source_rel IN ({placeholders})"
            try:
                rows = conn.execute(query, chunk).fetchall()
            except sqlite3.Error:
                continue
            for row in rows:
                if row and row[0]:
                    present.add(str(row[0]))
    finally:
        conn.close()
    return present


def iter_sqlite_jsonl_rows(
    *,
    sqlite_path: Path,
    source_rels: Sequence[str],
    table: str = "jsonl_records",
) -> Iterator[Dict[str, Any]]:
    if (not source_rels) or (not sqlite_path.exists()):
        return

    try:
        conn = _connect_readonly(sqlite_path)
    except sqlite3.Error:
        return
    try:
        for chunk in _chunked(list(source_rels), size=64):
            placeholders = ",".join("?" for _ in chunk)
            query = (
                f"SELECT payload_json FROM {table} "
                f"WHERE source_rel IN ({placeholders}) "
                f"ORDER BY source_rel, line_no"
            )
            try:
                rows = conn.execute(query, chunk)
            except sqlite3.Error:
                continue
            for (payload_json,) in rows:
                try:
                    obj = _json_loads(payload_json)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    finally:
        conn.close()


def iter_sqlite_jsonl_rows_by_like_patterns(
    *,
    sqlite_path: Path,
    like_patterns: Sequence[str],
    table: str = "jsonl_records",
) -> Iterator[Dict[str, Any]]:
    if (not like_patterns) or (not sqlite_path.exists()):
        return

    normalized = [str(pattern).strip() for pattern in like_patterns if str(pattern).strip()]
    if not normalized:
        return

    try:
        conn = _connect_readonly(sqlite_path)
    except sqlite3.Error:
        return
    try:
        seen_patterns: Set[str] = set()
        for chunk in _chunked(normalized, size=24):
            deduped_chunk: List[str] = []
            for pattern in chunk:
                if pattern in seen_patterns:
                    continue
                seen_patterns.add(pattern)
                deduped_chunk.append(pattern)
            if not deduped_chunk:
                continue
            clauses: List[str] = []
            params: List[str] = []
            for pattern in deduped_chunk:
                prefix = _literal_like_prefix(pattern)
                upper = _prefix_upper_bound(prefix)
                if prefix and upper:
                    clauses.append("(source_rel >= ? AND source_rel < ? AND source_rel LIKE ?)")
                    params.extend([prefix, upper, pattern])
                else:
                    clauses.append("source_rel LIKE ?")
                    params.append(pattern)
            where = " OR ".join(clauses)
            query = (
                f"SELECT payload_json FROM {table} INDEXED BY idx_jsonl_records_source_rel_line "
                f"WHERE {where} "
                f"ORDER BY source_rel, line_no"
            )
            try:
                rows = conn.execute(query, params)
            except sqlite3.Error:
                fallback_where = " OR ".join("source_rel LIKE ?" for _ in deduped_chunk)
                fallback_query = (
                    f"SELECT payload_json FROM {table} "
                    f"WHERE {fallback_where} "
                    f"ORDER BY source_rel, line_no"
                )
                try:
                    rows = conn.execute(fallback_query, deduped_chunk)
                except sqlite3.Error:
                    continue
            for (payload_json,) in rows:
                try:
                    obj = _json_loads(payload_json)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    finally:
        conn.close()


def split_paths_by_sqlite_coverage(
    *,
    project_root: Path,
    paths: Sequence[Path],
    sqlite_path: Optional[Path],
    table: str = "jsonl_records",
) -> Tuple[List[str], List[Path]]:
    if sqlite_path is None or (not sqlite_path.exists()) or (not paths):
        return [], list(paths)

    source_rels = source_rels_for_paths(project_root, paths)
    if not source_rels:
        return [], list(paths)

    try:
        present = source_rels_present_in_sqlite(sqlite_path=sqlite_path, source_rels=source_rels, table=table)
    except sqlite3.Error:
        return [], list(paths)
    missing_paths: List[Path] = []
    sql_source_rels: List[str] = []

    for path in paths:
        try:
            source_rel = source_rel_for_path(project_root, path)
        except Exception:
            missing_paths.append(path)
            continue
        if source_rel in present:
            sql_source_rels.append(source_rel)
        else:
            missing_paths.append(path)
    return sql_source_rels, missing_paths
