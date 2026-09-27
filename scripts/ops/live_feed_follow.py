#!/usr/bin/env python3
"""Read-only, bounded livefeed follower with periodic source rediscovery."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import stat
import sys
import time
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.storage_router import inspect_storage_path
from scripts.ops.long_runtime_common import run_bounded_process_group

MAX_READ = 256 * 1024
MAX_LINE = 32 * 1024


@dataclass
class Cursor:
    identity: tuple[int, int]
    offset: int
    pending: bytes = b""


class Follower:
    def __init__(self):
        self.cursors: dict[Path, Cursor] = {}

    def read(self, path: Path, *, initialize: bool = False) -> bytes:
        route = inspect_storage_path(path)
        if route.get("status") != "present":
            return b""
        try:
            with os.fdopen(
                os.open(route["resolved_path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
            ) as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode):
                    return b""
                identity = (info.st_dev, info.st_ino)
                cursor = self.cursors.get(path)
                if (
                    cursor is None
                    or cursor.identity != identity
                    or info.st_size < cursor.offset
                ):
                    cursor = Cursor(identity, info.st_size if initialize else 0)
                    self.cursors[path] = cursor
                handle.seek(cursor.offset)
                chunk = handle.read(MAX_READ)
                cursor.offset += len(chunk)
        except OSError:
            return b""
        combined = cursor.pending + chunk
        last = combined.rfind(b"\n")
        output = combined[: last + 1] if last >= 0 else b""
        cursor.pending = combined[last + 1 :]
        if len(cursor.pending) > MAX_LINE:
            output += cursor.pending[:MAX_LINE] + b" [livefeed line truncated]\n"
            cursor.pending = b""
        return output


def discover(command: list[str]) -> list[Path] | None:
    result = run_bounded_process_group(command, cwd=ROOT, timeout_seconds=10)
    if result["rc"] != 0 or result["timed_out"]:
        return None
    return list(
        dict.fromkeys(Path(value) for value in result["stdout"].split("\0") if value)
    )[:512]


def main() -> int:
    # Raising through the bounded discovery runner lets it reap its owned child.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--rediscover-seconds", type=int, default=60)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    follower = Follower()
    paths = [Path(value) for value in args.path][:512]
    for path in paths:
        follower.read(path, initialize=True)
    next_discovery = time.monotonic() + max(15, args.rediscover_seconds)
    try:
        while True:
            if time.monotonic() >= next_discovery:
                selected = discover(command) if command else None
                if selected is not None:
                    # Drain the previous selection before switching; unchanged paths retain offsets.
                    for path in paths:
                        sys.stdout.buffer.write(follower.read(path))
                    paths = selected
                    follower.cursors = {
                        path: cursor
                        for path, cursor in follower.cursors.items()
                        if path in paths
                    }
                next_discovery = time.monotonic() + max(15, args.rediscover_seconds)
            for path in paths:
                sys.stdout.buffer.write(follower.read(path))
            sys.stdout.buffer.flush()
            time.sleep(1)
    except (BrokenPipeError, KeyboardInterrupt):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
