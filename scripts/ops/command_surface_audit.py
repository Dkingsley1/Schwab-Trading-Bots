"""Non-executing inventory audit for the existing native command-validity owner."""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shlex
import stat
import time
from collections import Counter
from pathlib import Path

from core.storage_router import inspect_storage_path
from scripts.ops import commands_hygiene_bot as source
from scripts.ops.long_runtime_common import iso_now, run_bounded_process_group

SCRIPT = re.compile(r"(?<![\w/])(?:\./)?(scripts/[A-Za-z0-9_./-]+\.(?:py|sh))")
DISPATCH = re.compile(r"^  ([A-Za-z0-9_|-]+)\)\s*$", re.MULTILINE)
MAX_FILE_BYTES = 4 * 1024**2
MAX_TOTAL_BYTES = 64 * 1024**2


def _read_local(root: Path, relative: str) -> str:
    if inspect_storage_path(root).get("status") not in {"present", "missing"}:
        raise ValueError("protected_or_unavailable_project_root")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("outside_source_scope")
    current = root
    for part in path.parts:
        current /= part
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError("symlink_source_not_inspected")
    fd = os.open(current, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ValueError("source_file_budget_or_type")
        data = handle.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("source_file_budget_or_type")
    return data.decode("utf-8")


def _routes(text: str) -> dict[str, str]:
    matches = list(DISPATCH.finditer(text))
    routes = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        for alias in match.group(1).split("|"):
            routes[alias] = text[match.end() : end]
    return routes


def _subcommands(code: str) -> list[str]:
    result = []
    for line in code.replace("\\\n", " ").splitlines():
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        for index, token in enumerate(tokens[:-1]):
            if token in {"./scripts/ops/opsctl.sh", "scripts/ops/opsctl.sh"}:
                result.append(tokens[index + 1])
    return sorted(set(result))


def build_payload(project_root: Path, *, budget_seconds: float = 30) -> dict:
    started = time.monotonic()
    deadline = started + max(1, min(float(budget_seconds), 60))
    root = project_root.absolute()
    cache: dict[str, dict] = {}
    total_bytes = 0

    def inspect(relative: str) -> dict:
        nonlocal total_bytes
        if relative in cache:
            return cache[relative]
        row = {"path": relative, "status": "unverified", "sha256": "", "issues": []}
        cache[relative] = row
        if time.monotonic() >= deadline or total_bytes >= MAX_TOTAL_BYTES:
            row["issues"] = ["audit_budget_exhausted"]
            return row
        try:
            text = _read_local(root, relative)
            total_bytes += len(text.encode())
            row["sha256"] = hashlib.sha256(text.encode()).hexdigest()
            if relative.endswith(".py"):
                ast.parse(text, filename=relative)
            elif relative.endswith(".sh"):
                result = run_bounded_process_group(
                    ["/bin/zsh", "-f", "-n", str(root / relative)],
                    cwd=root,
                    timeout_seconds=max(1, min(2, int(deadline - time.monotonic()))),
                )
                if result["timed_out"]:
                    row["issues"] = ["syntax_check_timeout"]
                    return row
                if result["rc"]:
                    raise ValueError("shell_syntax_invalid")
            row["status"] = "static_pass"
        except (OSError, ValueError, SyntaxError, UnicodeError) as exc:
            row["status"] = "blocked"
            row["issues"] = [f"{type(exc).__name__}:{exc}"]
        return row

    try:
        commands = _read_local(root, "COMMANDS.md")
        opsctl = _read_local(root, "scripts/ops/opsctl.sh")
    except (OSError, ValueError, UnicodeError) as exc:
        return {
            "timestamp_utc": iso_now(),
            "ok": False,
            "overall_status": "blocked",
            "audit_mode": "non_executing",
            "issues": [str(exc)],
            "command_rows": [],
            "metrics": {"entry_count": 0, "blocked_entry_count": 1},
        }
    _, sections = source._parse_commands_sections(commands)
    routes = _routes(opsctl)
    contract = source.build_command_contract(root)
    expected_hash = contract["contract_hash"]
    match = re.search(r"Command contract hash:\s*`([0-9a-f]{64})`", commands)
    hash_mismatch = bool(
        "This file is generated from the curated operator inventory" in commands
        and (not match or match.group(1) != expected_hash)
    )
    rows = []
    for section in sections:
        for entry in section.get("entries", []):
            code = source._extract_first_code_block(entry.get("lines", []))
            title = str(entry.get("title") or "")
            aliases = _subcommands(code)
            paths = set(SCRIPT.findall(code))
            issues = [] if code.strip() else ["missing_code_block"]
            unresolved = []
            for alias in aliases:
                if alias not in routes:
                    issues.append(f"opsctl_dispatch_missing:{alias}")
                    continue
                body = (
                    routes[alias]
                    .replace("$PROJECT_ROOT/", "")
                    .replace("${PROJECT_ROOT}/", "")
                )
                targets = set(SCRIPT.findall(body))
                paths.update(targets)
                if not targets:
                    unresolved.append(alias)
            checks = [inspect(path) for path in sorted(paths)]
            issues.extend(
                f"{check['path']}:{issue}"
                for check in checks
                if check["status"] == "blocked"
                for issue in check["issues"]
            )
            incomplete = not paths or any(
                check["status"] == "unverified" for check in checks
            )
            status = (
                "blocked" if issues else "unverified" if incomplete else "static_pass"
            )
            rows.append(
                {
                    "title": title,
                    "section": section.get("heading", ""),
                    "entry_id": hashlib.sha256(
                        (title + "\n" + code).encode()
                    ).hexdigest(),
                    "validation_status": status,
                    "issues": issues,
                    "opsctl_subcommands": aliases,
                    "implementation_paths": sorted(paths),
                    "inline_or_dynamic_routes": unresolved,
                    "purpose_status": "declared_not_proven" if title else "missing",
                    "verification_scope": "source_presence_and_syntax_only",
                    "functional_status": "not_executed_requires_isolated_test_or_supervised_evidence",
                    "exact_arguments_verified": False,
                    "execution_authorized": False,
                    "next_action": (
                        "repair_source_or_curated_inventory"
                        if issues
                        else "supply_isolated_functional_test_or_supervised_receipt"
                    ),
                    "snippet_sha256": hashlib.sha256(code.strip().encode()).hexdigest(),
                }
            )
    counts = Counter(row["snippet_sha256"] for row in rows)
    duplicates = [
        {
            "snippet_sha256": digest,
            "titles": [row["title"] for row in rows if row["snippet_sha256"] == digest],
        }
        for digest, count in counts.items()
        if count > 1
    ]
    blocked = sum(row["validation_status"] == "blocked" for row in rows)
    unverified = sum(row["validation_status"] == "unverified" for row in rows)
    overall = (
        "blocked"
        if blocked or hash_mismatch or not rows
        else "degraded" if unverified else "ready"
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall != "blocked",
        "overall_status": overall,
        "audit_mode": "non_executing",
        "status_scope": "structural_audit_only_not_functional_certification",
        "commands_sha256": hashlib.sha256(commands.encode()).hexdigest(),
        "command_contract": {
            "expected_hash": expected_hash,
            "hash_mismatch": hash_mismatch,
        },
        "command_rows": rows,
        "source_checks": list(cache.values()),
        "duplicate_snippet_groups": duplicates,
        "authority": {
            "execute_documented_commands": False,
            "rewrite_contracts": False,
            "clear_halts": False,
            "delete_data": False,
            "enable_trading": False,
        },
        "metrics": {
            "entry_count": len(rows),
            "blocked_entry_count": blocked,
            "unverified_entry_count": unverified,
            "static_pass_count": len(rows) - blocked - unverified,
            "functionally_verified_entry_count": 0,
            "functional_evidence_gap_count": len(rows),
            "unique_source_count": len(cache),
            "duplicate_group_count": len(duplicates),
            "contract_hash_mismatch_count": int(hash_mismatch),
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "recommended_actions": [
            "repair blocked routes in the owning source",
            "review duplicate purpose before removing commands",
            "functional tests must not submit orders, delete retained data, or restart the live stack",
        ],
    }
