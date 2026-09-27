#!/usr/bin/env python3
"""Match live processes by argv tokens instead of flat command substrings."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessRow:
    pid: int
    stat: str
    command: str


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _token_matches(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    return "/" in expected and actual.endswith(expected)


def command_matches_pattern(command: str, pattern: str) -> bool:
    command_tokens = _split_command(str(command or "").strip())
    pattern_tokens = _split_command(str(pattern or "").strip())
    if not command_tokens or not pattern_tokens:
        return False

    command_index = 0
    for expected in pattern_tokens:
        while command_index < len(command_tokens):
            actual = command_tokens[command_index]
            command_index += 1
            if _token_matches(actual, expected):
                break
        else:
            return False
    return True


def parse_process_rows(raw: str) -> list[ProcessRow]:
    rows: list[ProcessRow] = []
    for line in raw.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid_raw, stat, command = parts
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        rows.append(ProcessRow(pid=pid, stat=stat, command=command))
    return rows


def matching_processes(pattern: str) -> list[ProcessRow]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,stat=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    own_pid = os.getpid()
    return [
        row
        for row in parse_process_rows(result.stdout or "")
        if row.pid != own_pid
        and not row.stat.startswith("T")
        and command_matches_pattern(row.command, pattern)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--count", action="store_true")
    output.add_argument("--first-pid", action="store_true")
    output.add_argument("--pids", action="store_true")
    args = parser.parse_args()

    matches = matching_processes(args.match)
    if args.count:
        print(len(matches))
    elif args.first_pid and matches:
        print(matches[0].pid)
    elif args.pids:
        for match in matches:
            print(match.pid)
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
