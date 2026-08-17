from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


DEFAULT_TAIL_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_TAIL_CHUNK_BYTES = 256 * 1024


def iter_tail_lines_reverse(path: Path, *, max_bytes: int, chunk_bytes: int = DEFAULT_TAIL_CHUNK_BYTES) -> Iterable[str]:
    limit = max(int(max_bytes), 1024)
    step = max(int(chunk_bytes), 1024)
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        position = fh.tell()
        if position <= 0:
            return
        remaining = min(position, limit)
        buffer = b""
        while remaining > 0:
            read_size = min(step, remaining)
            position -= read_size
            fh.seek(position)
            chunk = fh.read(read_size)
            if not chunk:
                break
            remaining -= read_size
            buffer = chunk + buffer
            parts = buffer.split(b"\n")
            buffer = parts[0]
            for raw in reversed(parts[1:]):
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    yield line
        tail = buffer.decode("utf-8", "replace").strip()
        if tail:
            yield tail


def count_tail_keyword(
    path: Path,
    keyword: str,
    *,
    tail_lines: int = 2000,
    max_bytes: int = DEFAULT_TAIL_MAX_BYTES,
    chunk_bytes: int = DEFAULT_TAIL_CHUNK_BYTES,
) -> int:
    wanted = str(keyword or "").strip().casefold()
    if not wanted or not path.exists():
        return 0

    count = 0
    seen = 0
    for line in iter_tail_lines_reverse(path, max_bytes=max_bytes, chunk_bytes=chunk_bytes):
        seen += 1
        if wanted in line.casefold():
            count += 1
        if seen >= max(int(tail_lines), 1):
            break
    return count
