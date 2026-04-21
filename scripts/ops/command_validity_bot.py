#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import commands_hygiene_bot as commands_src
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from . import commands_hygiene_bot as commands_src
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


COMMANDS_PATH = PROJECT_ROOT / "COMMANDS.md"
OPSCTL_PATH = PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"
RUNBOOK_PATH = PROJECT_ROOT / "scripts" / "runbook.sh"
PYTHON_BIN = Path(sys.executable)
MUTATING_KEYWORDS = (
    " start",
    " stop",
    " feed-refresh",
    " token-refresh-interactive",
    " --apply",
    " --set-global-halt",
    " --clear-global-halt",
    " storage-switch-",
    " storage-safe-eject",
    " macro-auto-start",
)
SHELL_CONTROL_TOKENS = {
    "|",
    "||",
    "&&",
    ";",
    "do",
    "done",
    "then",
    "fi",
    "else",
    "elif",
    "case",
    "esac",
    "while",
    "for",
    "if",
}
SHELL_FRAGMENT_PREFIXES = ("|", "||", "&&", ";", "-", "\\(", "\\)", "(", ")", "{", "}")
OPERATOR_GATED_COMMANDS = {
    "caffeinate",
    "diskutil",
    "kill",
    "launchctl",
    "nohup",
    "open",
    "osascript",
    "pkill",
    "rm",
}
OPERATOR_GATED_FRAGMENTS = (
    " rm ",
    " -delete",
    " -exec rm",
    " bootout ",
    " bootstrap ",
    " disable ",
    " enable ",
    " kickstart ",
    " unload ",
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _run_result(cmd: list[str], *, cwd: Path, timeout_sec: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    return {
        "cmd": list(cmd),
        "rc": rc,
        "timed_out": timed_out,
        "ok": rc == 0 and not timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
    }


def _opsctl_dispatch_aliases(text: str) -> set[str]:
    aliases: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^\s{2}([A-Za-z0-9_|-]+)\)", line)
        if not match:
            continue
        for raw in match.group(1).split("|"):
            alias = raw.strip()
            if alias:
                aliases.add(alias)
    return aliases


def _opsctl_help_aliases(help_text: str) -> set[str]:
    aliases: set[str] = set()
    for raw in help_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("Usage:"):
            continue
        if not re.match(r"^[a-z0-9][a-z0-9_|-]*(?:\s|\[|$)", line):
            continue
        token = line.split()[0]
        for alias in token.split("|"):
            alias = alias.strip()
            if alias:
                aliases.add(alias)
    return aliases


def _extract_runtime_python_path(line: str) -> Path | None:
    match = re.search(r"zsh\s+(\./scripts/ops/runtime_python\.sh)", line)
    if not match:
        return None
    return Path(match.group(1))


def _parse_tokens(line: str) -> list[str]:
    try:
        return shlex.split(line)
    except Exception:
        return []


def _logical_lines(code_block: str) -> list[str]:
    logical: list[str] = []
    buffer = ""
    for raw in code_block.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if buffer:
            buffer = f"{buffer} {stripped}"
        else:
            buffer = stripped
        if buffer.endswith("\\"):
            buffer = buffer[:-1].rstrip()
            continue
        logical.append(buffer)
        buffer = ""
    if buffer:
        logical.append(buffer)
    return logical


def _is_shell_fragment(line: str, tokens: list[str]) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    if not tokens:
        return False
    first = tokens[0]
    if first in SHELL_CONTROL_TOKENS:
        return True
    if any(first.startswith(prefix) for prefix in SHELL_FRAGMENT_PREFIXES):
        return True
    return False


def _script_paths_from_line(line: str) -> list[Path]:
    paths: list[Path] = []
    runtime_python = _extract_runtime_python_path(line)
    if runtime_python is not None:
        paths.append(runtime_python)

    tokens = _parse_tokens(line)
    if not tokens:
        return paths
    if tokens[0] == "cd" and len(tokens) >= 2:
        return paths

    for token in tokens:
        if token.startswith("./scripts/") or token.startswith("scripts/"):
            candidate = Path(token)
            if candidate.suffix in {".py", ".sh"}:
                paths.append(candidate)
    return paths


def _validate_script(path: Path, *, project_root: Path) -> list[str]:
    issues: list[str] = []
    full_path = path if path.is_absolute() else project_root / path
    if not full_path.exists():
        return [f"missing:{path.as_posix()}"]
    if full_path.suffix == ".sh":
        syntax = _run_result(["zsh", "-n", str(full_path)], cwd=project_root, timeout_sec=15)
        if not bool(syntax.get("ok", False)):
            issues.append(f"syntax_error:{path.as_posix()}")
    elif full_path.suffix == ".py":
        compile_result = _run_result([str(PYTHON_BIN), "-m", "py_compile", str(full_path)], cwd=project_root, timeout_sec=15)
        if not bool(compile_result.get("ok", False)):
            issues.append(f"syntax_error:{path.as_posix()}")
    return issues


def _line_safety(line: str) -> str:
    lowered = f" {line.strip().lower()}"
    if any(keyword in lowered for keyword in MUTATING_KEYWORDS):
        return "operator_gated"
    if "phone-feed" in lowered:
        return "operator_gated"
    tokens = _parse_tokens(line.strip())
    if tokens and tokens[0] in OPERATOR_GATED_COMMANDS:
        return "operator_gated"
    if any(fragment in lowered for fragment in OPERATOR_GATED_FRAGMENTS):
        return "operator_gated"
    return "safe_static"


def _validate_entry(
    *,
    title: str,
    code_block: str,
    project_root: Path,
    opsctl_dispatch_aliases: set[str],
    opsctl_help_aliases: set[str],
) -> dict[str, Any]:
    lines = _logical_lines(code_block)
    issues: list[str] = []
    safety_flags: list[str] = []

    for line in lines:
        stripped = line.strip()
        safety_flags.append(_line_safety(stripped))
        if re.match(r"^[A-Z_][A-Z0-9_]*=", stripped):
            for script_path in _script_paths_from_line(stripped):
                issues.extend(_validate_script(script_path, project_root=project_root))
            continue
        if stripped.startswith("cd "):
            target = Path(stripped[3:].strip())
            if not target.exists():
                issues.append(f"missing:{target}")
            continue

        tokens = _parse_tokens(stripped)
        if _is_shell_fragment(stripped, tokens):
            continue
        if tokens[:2] == ["./scripts/ops/opsctl.sh", "help"] or tokens[:2] == ["scripts/ops/opsctl.sh", "help"]:
            continue

        if tokens and tokens[0] in {"./scripts/ops/opsctl.sh", "scripts/ops/opsctl.sh"}:
            if len(tokens) < 2:
                issues.append("opsctl_subcommand_missing")
            else:
                subcmd = tokens[1]
                if subcmd not in opsctl_dispatch_aliases:
                    issues.append(f"opsctl_dispatch_missing:{subcmd}")
                if subcmd not in opsctl_help_aliases:
                    issues.append(f"opsctl_help_missing:{subcmd}")

        for script_path in _script_paths_from_line(stripped):
            issues.extend(_validate_script(script_path, project_root=project_root))

        if tokens and tokens[0] not in {"cd", "./scripts/ops/opsctl.sh", "scripts/ops/opsctl.sh", "$PY", '"$PY"'}:
            command_name = tokens[0]
            if not command_name.startswith(("./", "scripts/")) and shutil.which(command_name) is None:
                issues.append(f"external_command_missing:{command_name}")

    safety = "operator_gated" if "operator_gated" in safety_flags else "safe_static"
    validation_status = "blocked" if issues else ("operator_gated" if safety == "operator_gated" else "ready")
    return {
        "title": title,
        "validation_status": validation_status,
        "safety": safety,
        "issues": ordered_unique(issues),
        "command_excerpt": lines[0] if lines else "",
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    timeout_sec: int = 45,
) -> dict[str, Any]:
    commands_path = project_root / "COMMANDS.md"
    opsctl_path = project_root / "scripts" / "ops" / "opsctl.sh"
    runbook_path = project_root / "scripts" / "runbook.sh"

    hygiene = _run_result(
        [
            str(PYTHON_BIN),
            str(project_root / "scripts" / "ops" / "commands_hygiene_bot.py"),
            "--project-root",
            str(project_root),
            "--apply" if apply else "--json",
            "--json",
        ]
        if apply
        else [
            str(PYTHON_BIN),
            str(project_root / "scripts" / "ops" / "commands_hygiene_bot.py"),
            "--project-root",
            str(project_root),
            "--json",
        ],
        cwd=project_root,
        timeout_sec=max(int(timeout_sec), 15),
    )

    commands_text = _read_text(commands_path)
    opsctl_text = _read_text(opsctl_path)
    preamble, sections = commands_src._parse_commands_sections(commands_text)
    _ = preamble

    opsctl_syntax = _run_result(["zsh", "-n", str(opsctl_path)], cwd=project_root, timeout_sec=15)
    runbook_syntax = _run_result(["zsh", "-n", str(runbook_path)], cwd=project_root, timeout_sec=15) if runbook_path.exists() else {
        "ok": False,
        "rc": 1,
        "timed_out": False,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    opsctl_help = _run_result(["zsh", str(opsctl_path), "help"], cwd=project_root, timeout_sec=15)
    opsctl_dispatch_aliases = _opsctl_dispatch_aliases(opsctl_text)
    opsctl_help_aliases = _opsctl_help_aliases(str(opsctl_help.get("stdout") or ""))

    rows: list[dict[str, Any]] = []
    for section in sections:
        for entry in list(section.get("entries") or []):
            title = str(entry.get("title") or "").strip()
            code_block = commands_src._extract_first_code_block(list(entry.get("lines") or []))
            if not code_block:
                rows.append(
                    {
                        "title": title,
                        "validation_status": "blocked",
                        "safety": "safe_static",
                        "issues": ["missing_code_block"],
                        "command_excerpt": "",
                    }
                )
                continue
            rows.append(
                _validate_entry(
                    title=title,
                    code_block=code_block,
                    project_root=project_root,
                    opsctl_dispatch_aliases=opsctl_dispatch_aliases,
                    opsctl_help_aliases=opsctl_help_aliases,
                )
            )

    blocked_rows = [row for row in rows if str(row.get("validation_status") or "") == "blocked"]
    operator_gated_rows = [row for row in rows if str(row.get("validation_status") or "") == "operator_gated"]
    smoke_failures = ordered_unique(
        [
            "opsctl_syntax_invalid" if not bool(opsctl_syntax.get("ok", False)) else "",
            "opsctl_help_failed" if not bool(opsctl_help.get("ok", False)) else "",
            "runbook_syntax_invalid" if not bool(runbook_syntax.get("ok", False)) else "",
        ]
    )

    overall_status = "ready"
    if blocked_rows or smoke_failures:
        overall_status = "blocked"
    elif operator_gated_rows:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "use command-validity after changing opsctl aliases or command docs so COMMANDS.md stays executable in practice",
            "keep commands-hygiene in apply mode before command-validity when you want the docs normalized and then re-verified",
            "review blocked command entries before relying on them during live incident handling" if blocked_rows else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "apply_requested": bool(apply),
        "commands_path": str(commands_path),
        "command_rows": rows,
        "smoke": {
            "opsctl_syntax": opsctl_syntax,
            "opsctl_help": opsctl_help,
            "runbook_syntax": runbook_syntax,
        },
        "commands_hygiene": hygiene,
        "metrics": {
            "entry_count": len(rows),
            "blocked_entry_count": len(blocked_rows),
            "operator_gated_entry_count": len(operator_gated_rows),
            "smoke_failure_count": len(smoke_failures),
        },
        "operator_followups": [f"repair {row['title']} because it no longer resolves cleanly from COMMANDS.md" for row in blocked_rows[:6]],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate that documented COMMANDS.md snippets still resolve to real scripts, aliases, and syntactically valid entrypoints.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=int(os.getenv("COMMAND_VALIDITY_TIMEOUT_SECONDS", "45")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        apply=bool(args.apply),
        timeout_sec=int(args.timeout_sec),
    )
    out_file = Path(args.out_file).expanduser() if args.out_file else project_root / "governance" / "health" / "command_validity_latest.json"
    write_payload(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "command_validity_bot "
            f"overall_status={payload.get('overall_status', '')} "
            f"blocked_entries={int((payload.get('metrics') or {}).get('blocked_entry_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
