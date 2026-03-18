import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"


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


def source_rels_present_in_sqlite(
    *,
    sqlite_path: Path,
    source_rels: Sequence[str],
    table: str = "jsonl_records",
) -> Set[str]:
    if (not source_rels) or (not sqlite_path.exists()):
        return set()

    present: Set[str] = set()
    conn = sqlite3.connect(str(sqlite_path))
    try:
        for chunk in _chunked(list(source_rels)):
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT DISTINCT source_rel FROM {table} WHERE source_rel IN ({placeholders})"
            rows = conn.execute(query, chunk).fetchall()
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

    conn = sqlite3.connect(str(sqlite_path))
    try:
        for chunk in _chunked(list(source_rels)):
            placeholders = ",".join("?" for _ in chunk)
            query = (
                f"SELECT payload_json FROM {table} "
                f"WHERE source_rel IN ({placeholders}) "
                f"ORDER BY source_rel, line_no"
            )
            for (payload_json,) in conn.execute(query, chunk):
                try:
                    obj = json.loads(payload_json)
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

    present = source_rels_present_in_sqlite(sqlite_path=sqlite_path, source_rels=source_rels, table=table)
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
