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
    " livefeed-refresh",
    " live-feed-refresh",
    " token-refresh",
    " token-refresh-interactive",
    " --apply",
    " --set-global-halt",
    " --clear-global-halt",
    " --clear-blockers",
    " --auto-clear",
    " storage-switch-",
    " storage-safe-eject",
    " retrain-force",
    " retrain-orchestrate",
    " run_mode_switchboard.py",
    " daily_log_refresh.sh",
    " build_one_numbers_report.py",
    " report-pdfs",
    " showcase-refresh",
    " system-explainers",
    " dashboard-refresh",
    " paper-performance",
    " incident-report",
    " bot-stack-report",
    " open_report_artifact.sh",
    " macro-auto-start",
)
OPERATOR_GATED_OPSCTL_SUBCOMMANDS = {
    "active-bot-report",
    "bot-stack-report",
    "coinbase-futures-tail",
    "coinbase-tail",
    "command-validity",
    "commands-verify",
    "dashboard-refresh",
    "feed",
    "feed-refresh",
    "futures-tail",
    "fx-tail",
    "incident-report",
    "live-feed-refresh",
    "livefeed-refresh",
    "main-tail",
    "macro-auto-start",
    "macro-auto-stop",
    "macro-media-ingest",
    "macro-replay",
    "paper-performance",
    "phone-feed",
    "report-pdfs",
    "retrain-force-full",
    "retrain-force-targeted",
    "retrain-orchestrate",
    "schwab-futures-tail",
    "schwab-tail",
    "showcase-refresh",
    "start",
    "start-live",
    "start-sim",
    "stop",
    "storage-disaster-recovery",
    "storage-pressure-clearance",
    "storage-safe-eject",
    "storage-switch-external",
    "storage-switch-local",
    "system-explainers",
    "system-explainer-docs",
    "token-refresh",
    "token-refresh-interactive",
}
OPEN_REPORT_ARTIFACT_KINDS = {
    "botstack",
    "bundle",
    "calibration",
    "correlation",
    "crash",
    "daily-auto-verify",
    "daily-ops",
    "daily-runtime",
    "explainability",
    "expansions",
    "framework",
    "incident",
    "incident-packet",
    "macro",
    "modelcard",
    "one-numbers",
    "one-numbers-csv",
    "paper",
    "posttrade",
    "quant",
    "report-catalog",
    "replay",
    "sendout",
    "sentiment",
    "special",
    "source",
    "state-snapshot",
    "strategy-attribution",
    "strategy-inventory",
    "summary",
    "system-overview",
    "timeline",
    "training",
    "unified",
}
OPSCTL_ROUTE_PROBES = {
    "bot-stack-report": {"args": ["--help"], "expected_rcs": {0}},
    "commands-hygiene": {"args": ["--help"], "expected_rcs": {0}},
    "command-validity": {"args": ["--help"], "expected_rcs": {0}},
    "coinbase-futures-tail": {"args": ["--help"], "expected_rcs": {0}},
    "coinbase-tail": {"args": ["--help"], "expected_rcs": {0}},
    "dashboard-refresh": {"args": ["--help"], "expected_rcs": {0}},
    "feed": {"args": ["--help"], "expected_rcs": {0}},
    "feed-refresh": {"args": ["--dry-run"], "expected_rcs": {0}},
    "futures-tail": {"args": ["--help"], "expected_rcs": {0}},
    "fx-tail": {"args": ["--help"], "expected_rcs": {0}},
    "incident-report": {"args": ["--help"], "expected_rcs": {0}},
    "live-feed-refresh": {"args": ["--dry-run"], "expected_rcs": {0}},
    "livefeed-refresh": {"args": ["--dry-run"], "expected_rcs": {0}},
    "macro-auto-start": {"args": ["--dry-run"], "expected_rcs": {0}},
    "health-gates": {"args": ["--help"], "expected_rcs": {0}},
    "storage-pressure-clearance": {"args": ["--help"], "expected_rcs": {0}},
    "main-tail": {"args": ["--help"], "expected_rcs": {0}},
    "paper-performance": {"args": ["--help"], "expected_rcs": {0}},
    "phone-feed": {"args": ["--help"], "expected_rcs": {0}},
    "production-hardening-watch": {"args": ["--help"], "expected_rcs": {0}},
    "production-excellence": {"args": ["--help"], "expected_rcs": {0}},
    "production-quality": {"args": ["--help"], "expected_rcs": {0}},
    "production-quality-slo": {"args": ["--help"], "expected_rcs": {0}},
    "report-quality-guard": {"args": ["--help"], "expected_rcs": {0}},
    "report-pdfs": {"args": ["--help"], "expected_rcs": {0}},
    "live-order-ledger": {"args": ["--help"], "expected_rcs": {0}},
    "strategy-inventory": {"args": ["--help"], "expected_rcs": {0}},
    "retrain-force-full": {"args": ["--help"], "expected_rcs": {0}},
    "retrain-orchestrate": {"args": ["--help"], "expected_rcs": {0}},
    "schwab-futures-tail": {"args": ["--help"], "expected_rcs": {0}},
    "schwab-auth-supervisor": {"args": ["--help"], "expected_rcs": {0}},
    "schwab-tail": {"args": ["--help"], "expected_rcs": {0}},
    "service-control-plane": {"args": ["--help"], "expected_rcs": {0}},
    "showcase-refresh": {"args": ["--help"], "expected_rcs": {0}},
    "start": {"args": ["--dry-run"], "expected_rcs": {0, 2}},
    "start-live": {"args": ["--dry-run"], "expected_rcs": {0, 2}},
    "start-sim": {"args": ["--dry-run"], "expected_rcs": {0, 2}},
    "stop": {"args": ["--dry-run"], "expected_rcs": {0}},
    "storage-disaster-recovery": {"args": ["--help"], "expected_rcs": {0}},
    "storage-safe-eject": {"args": ["--dry-run"], "expected_rcs": {0}},
    "storage-switch-external": {"args": ["--dry-run"], "expected_rcs": {0}},
    "storage-switch-local": {"args": ["--dry-run"], "expected_rcs": {0}},
    "system-explainers": {"args": ["--help"], "expected_rcs": {0}},
    "token-refresh": {"args": ["--help"], "expected_rcs": {0}},
    "token-refresh-interactive": {"args": ["--help"], "expected_rcs": {0}},
}
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


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


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
    if tokens and tokens[0] in {"./scripts/ops/opsctl.sh", "scripts/ops/opsctl.sh"} and len(tokens) >= 2:
        if tokens[1] in OPERATOR_GATED_OPSCTL_SUBCOMMANDS:
            return "operator_gated"
    if tokens and tokens[0] in OPERATOR_GATED_COMMANDS:
        return "operator_gated"
    if any(fragment in lowered for fragment in OPERATOR_GATED_FRAGMENTS):
        return "operator_gated"
    return "safe_static"


def _open_target_issues(tokens: list[str], *, project_root: Path) -> list[str]:
    if not tokens or tokens[0] != "open":
        return []
    if len(tokens) < 2:
        return ["open_target_missing"]
    target = Path(tokens[1]).expanduser()
    if not target.is_absolute():
        target = project_root / target
    if not target.exists():
        return [f"open_target_missing:{target}"]
    return []


def _open_report_artifact_issues(tokens: list[str]) -> list[str]:
    if not tokens or tokens[0] not in {"./scripts/ops/open_report_artifact.sh", "scripts/ops/open_report_artifact.sh"}:
        return []
    report_args = [token for token in tokens[1:] if not token.startswith("-")]
    if not report_args:
        return ["open_report_artifact_kind_missing"]
    kind = report_args[-1]
    if kind not in OPEN_REPORT_ARTIFACT_KINDS:
        return [f"open_report_artifact_kind_unknown:{kind}"]
    return []


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

        issues.extend(_open_target_issues(tokens, project_root=project_root))
        issues.extend(_open_report_artifact_issues(tokens))

        for script_path in _script_paths_from_line(stripped):
            issues.extend(_validate_script(script_path, project_root=project_root))

        if tokens and tokens[0] not in {"cd", "./scripts/ops/opsctl.sh", "scripts/ops/opsctl.sh", "$PY", '"$PY"'}:
            command_name = tokens[0]
            if not command_name.startswith(("./", "scripts/")) and shutil.which(command_name) is None:
                issues.append(f"external_command_missing:{command_name}")

    safety = "operator_gated" if "operator_gated" in safety_flags else "safe_static"
    def is_soft_generated_open_issue(issue: str) -> bool:
        return (
            issue.startswith("open_target_missing:")
            and title.lower().startswith("open ")
            and "/exports/" in issue
        )

    blocking_issues = [issue for issue in issues if not is_soft_generated_open_issue(str(issue))]
    validation_status = "blocked" if blocking_issues else ("degraded" if issues else ("operator_gated" if safety == "operator_gated" else "ready"))
    return {
        "title": title,
        "validation_status": validation_status,
        "safety": safety,
        "issues": ordered_unique(issues),
        "command_excerpt": lines[0] if lines else "",
    }


def _safe_command_smokes(
    project_root: Path,
    *,
    timeout_sec: int,
    opsctl_dispatch_aliases: set[str],
    opsctl_help_aliases: set[str],
) -> list[dict[str, Any]]:
    checks = [
        {
            "name": "global_halt_status",
            "cmd": [str(PYTHON_BIN), str(project_root / "scripts" / "global_risk_killswitch.py"), "--status-only"],
            "expected_rcs": {0, 1, 2},
            "required_paths": [project_root / "scripts" / "global_risk_killswitch.py"],
            "validator": lambda result: isinstance(_json_from_text(str(result.get("stdout") or "")).get("clear_blockers"), list),
        },
        {
            "name": "global_halt_clear_blockers",
            "cmd": [str(PYTHON_BIN), str(project_root / "scripts" / "global_risk_killswitch.py"), "--clear-blockers", "--status-only"],
            "expected_rcs": {0, 1, 2},
            "required_paths": [project_root / "scripts" / "global_risk_killswitch.py"],
            "validator": lambda result: isinstance(_json_from_text(str(result.get("stdout") or "")).get("clear_blocker_refresh_attempts"), list),
        },
        {
            "name": "stack_start_dry_run",
            "cmd": ["zsh", str(project_root / "scripts" / "ops" / "opsctl.sh"), "start", "--dry-run"],
            "expected_rcs": {0, 2},
            "required_paths": [project_root / "scripts" / "ops" / "opsctl.sh", project_root / "scripts" / "ops" / "start_stack.sh"],
            "required_aliases": {"start"},
            "validator": lambda result: "stack_start_status=" in str(result.get("stdout") or ""),
        },
        {
            "name": "stack_stop_dry_run",
            "cmd": ["zsh", str(project_root / "scripts" / "ops" / "opsctl.sh"), "stop", "--dry-run"],
            "expected_rcs": {0},
            "required_paths": [project_root / "scripts" / "ops" / "opsctl.sh"],
            "required_aliases": {"stop"},
            "validator": lambda result: "stack_stop_status=ready_to_stop" in str(result.get("stdout") or ""),
        },
    ]
    rows: list[dict[str, Any]] = []
    for check in checks:
        required_paths = [Path(path) for path in list(check.get("required_paths") or [])]
        missing_required = [str(path) for path in required_paths if not path.exists()]
        if missing_required:
            rows.append(
                {
                    "name": str(check["name"]),
                    "cmd": list(check["cmd"]),
                    "expected_rcs": sorted(set(check["expected_rcs"])),
                    "rc": 0,
                    "timed_out": False,
                    "ok": True,
                    "skipped": True,
                    "skip_reason": "missing_required_paths",
                    "missing_required_paths": missing_required,
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
            )
            continue
        required_aliases = set(check.get("required_aliases") or set())
        if required_aliases and not required_aliases <= (set(opsctl_dispatch_aliases) | set(opsctl_help_aliases)):
            rows.append(
                {
                    "name": str(check["name"]),
                    "cmd": list(check["cmd"]),
                    "expected_rcs": sorted(set(check["expected_rcs"])),
                    "rc": 0,
                    "timed_out": False,
                    "ok": True,
                    "skipped": True,
                    "skip_reason": "missing_required_aliases",
                    "missing_required_aliases": sorted(required_aliases - (set(opsctl_dispatch_aliases) | set(opsctl_help_aliases))),
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
            )
            continue
        result = _run_result(check["cmd"], cwd=project_root, timeout_sec=max(int(timeout_sec), 15))
        validator = check["validator"]
        expected_rcs = set(check["expected_rcs"])
        rc = int(result.get("rc", 1))
        ok = rc in expected_rcs and not bool(result.get("timed_out", False)) and bool(validator(result))
        rows.append(
            {
                "name": str(check["name"]),
                "cmd": list(check["cmd"]),
                "expected_rcs": sorted(expected_rcs),
                "rc": rc,
                "timed_out": bool(result.get("timed_out", False)),
                "ok": bool(ok),
                "stdout_tail": str(result.get("stdout_tail") or ""),
                "stderr_tail": str(result.get("stderr_tail") or ""),
            }
        )
    return rows


def _json_from_text(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _commands_md_contract_hash(text: str) -> str:
    match = re.search(r"Command contract hash:\s*`([0-9a-f]{64})`", str(text or ""))
    return match.group(1) if match else ""


def _opsctl_subcommands_from_code_block(code_block: str) -> list[str]:
    subcommands: list[str] = []
    for line in _logical_lines(code_block):
        tokens = _parse_tokens(line.strip())
        if len(tokens) >= 2 and tokens[0] in {"./scripts/ops/opsctl.sh", "scripts/ops/opsctl.sh"}:
            subcommands.append(tokens[1])
    return ordered_unique(subcommands)


def _contract_dispatch_smokes(
    *,
    rows: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    project_root: Path,
    opsctl_dispatch_aliases: set[str],
    opsctl_help_aliases: set[str],
    timeout_sec: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    opsctl_path = project_root / "scripts" / "ops" / "opsctl.sh"
    seen: set[str] = set()
    smoke_rows: list[dict[str, Any]] = []
    probed_operator_rows: list[dict[str, Any]] = []

    title_by_subcmd: dict[str, str] = {}
    safety_by_subcmd: dict[str, str] = {}
    for validation_row, section in zip(rows, [entry for sec in sections for entry in list(sec.get("entries") or [])]):
        code_block = commands_src._extract_first_code_block(list(section.get("lines") or []))
        for subcmd in _opsctl_subcommands_from_code_block(code_block):
            title_by_subcmd.setdefault(subcmd, str(validation_row.get("title") or ""))
            safety_by_subcmd.setdefault(subcmd, str(validation_row.get("safety") or "safe_static"))

    for subcmd, spec in OPSCTL_ROUTE_PROBES.items():
        if subcmd not in title_by_subcmd or subcmd in seen:
            continue
        seen.add(subcmd)
        expected_rcs = set(spec.get("expected_rcs") or {0})
        cmd = ["zsh", str(opsctl_path), subcmd, *list(spec.get("args") or [])]
        result = _run_result(cmd, cwd=project_root, timeout_sec=max(int(timeout_sec), 15))
        rc = int(result.get("rc", 1))
        alias_ok = subcmd in opsctl_dispatch_aliases and subcmd in opsctl_help_aliases
        ok = alias_ok and rc in expected_rcs and not bool(result.get("timed_out", False))
        smoke_row = {
            "name": f"opsctl_route_probe:{subcmd}",
            "title": title_by_subcmd.get(subcmd, ""),
            "subcommand": subcmd,
            "safety": safety_by_subcmd.get(subcmd, "safe_static"),
            "cmd": cmd,
            "expected_rcs": sorted(expected_rcs),
            "rc": rc,
            "timed_out": bool(result.get("timed_out", False)),
            "ok": bool(ok),
            "alias_ok": bool(alias_ok),
            "stdout_tail": str(result.get("stdout_tail") or ""),
            "stderr_tail": str(result.get("stderr_tail") or ""),
        }
        smoke_rows.append(smoke_row)
        if smoke_row["safety"] == "operator_gated":
            probed_operator_rows.append(smoke_row)

    return smoke_rows, probed_operator_rows


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
    hygiene_payload = _json_from_text(str(hygiene.get("stdout") or ""))
    command_contract = commands_src.build_command_contract(project_root)
    expected_contract_hash = str(command_contract.get("contract_hash") or "")
    documented_contract_hash = _commands_md_contract_hash(commands_text)
    contract_hash_required = "This file is generated from the curated operator inventory" in commands_text
    contract_hash_mismatch = bool(contract_hash_required and expected_contract_hash and documented_contract_hash != expected_contract_hash)
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
    degraded_rows = [row for row in rows if str(row.get("validation_status") or "") == "degraded"]
    operator_gated_rows = [row for row in rows if str(row.get("validation_status") or "") == "operator_gated"]
    contract_dispatch_smokes, probed_operator_rows = _contract_dispatch_smokes(
        rows=rows,
        sections=sections,
        project_root=project_root,
        opsctl_dispatch_aliases=opsctl_dispatch_aliases,
        opsctl_help_aliases=opsctl_help_aliases,
        timeout_sec=max(int(timeout_sec), 15),
    )
    runtime_smoke_rows = _safe_command_smokes(
        project_root,
        timeout_sec=max(int(timeout_sec), 15),
        opsctl_dispatch_aliases=opsctl_dispatch_aliases,
        opsctl_help_aliases=opsctl_help_aliases,
    )
    runtime_smoke_failures = [row for row in runtime_smoke_rows if not bool(row.get("ok", False))]
    contract_dispatch_smoke_failures = [row for row in contract_dispatch_smokes if not bool(row.get("ok", False))]
    probed_operator_subcommands = {str(row.get("subcommand") or "") for row in probed_operator_rows}
    operator_gated_subcommands = set()
    for section in sections:
        for entry in list(section.get("entries") or []):
            code_block = commands_src._extract_first_code_block(list(entry.get("lines") or []))
            for subcmd in _opsctl_subcommands_from_code_block(code_block):
                if subcmd in OPERATOR_GATED_OPSCTL_SUBCOMMANDS:
                    operator_gated_subcommands.add(subcmd)
    unprobed_operator_gated_subcommands = sorted(operator_gated_subcommands - probed_operator_subcommands)
    hygiene_metrics = hygiene_payload.get("metrics") if isinstance(hygiene_payload.get("metrics"), dict) else {}
    hygiene_apply_results = hygiene_payload.get("apply_results") if isinstance(hygiene_payload.get("apply_results"), dict) else {}
    hygiene_failures = ordered_unique(
        [
            "commands_hygiene_duplicate_entries"
            if _safe_int(hygiene_metrics.get("duplicate_entry_count"), 0) > 0
            else "",
            "commands_hygiene_pending_changes"
            if (
                bool(hygiene_payload.get("commands_changed", False))
                and not bool(hygiene_apply_results.get("commands_md_written", False))
            )
            or (
                bool(hygiene_payload.get("runbook_changed", False))
                and not bool(hygiene_apply_results.get("runbook_written", False))
            )
            else "",
            "commands_contract_hash_mismatch" if contract_hash_mismatch else "",
        ]
    )
    smoke_failures = ordered_unique(
        [
            "opsctl_syntax_invalid" if not bool(opsctl_syntax.get("ok", False)) else "",
            "opsctl_help_failed" if not bool(opsctl_help.get("ok", False)) else "",
            "runbook_syntax_invalid" if not bool(runbook_syntax.get("ok", False)) else "",
        ]
    )

    overall_status = "ready"
    if blocked_rows or smoke_failures or runtime_smoke_failures or contract_dispatch_smoke_failures or hygiene_failures:
        overall_status = "blocked"
    elif degraded_rows or unprobed_operator_gated_subcommands:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "use command-validity after changing opsctl aliases or command docs so COMMANDS.md stays executable in practice",
            "keep commands-hygiene in apply mode before command-validity when you want the docs normalized and then re-verified",
            "review blocked command entries before relying on them during live incident handling" if blocked_rows else "",
            "repair halt or stack dry-run smoke failures before trusting the operator command surface" if runtime_smoke_failures else "",
            "repair exact opsctl route probe failures before trusting COMMANDS.md during live operations"
            if contract_dispatch_smoke_failures
            else "",
            "run commands-hygiene --apply to remove duplicate or stale generated command entries" if hygiene_failures else "",
            "add no-side-effect route probes for remaining operator-gated commands"
            if unprobed_operator_gated_subcommands
            else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "apply_requested": bool(apply),
        "commands_path": str(commands_path),
        "command_contract": {
            "expected_hash": expected_contract_hash,
            "documented_hash": documented_contract_hash,
            "hash_required": contract_hash_required,
            "hash_mismatch": contract_hash_mismatch,
            "entry_count": int(command_contract.get("entry_count") or 0),
        },
        "command_rows": rows,
        "smoke": {
            "opsctl_syntax": opsctl_syntax,
            "opsctl_help": opsctl_help,
            "runbook_syntax": runbook_syntax,
            "commands_hygiene_failures": hygiene_failures,
        },
        "runtime_smoke": runtime_smoke_rows,
        "contract_dispatch_smoke": contract_dispatch_smokes,
        "commands_hygiene": hygiene,
        "metrics": {
            "entry_count": len(rows),
            "blocked_entry_count": len(blocked_rows),
            "degraded_entry_count": len(degraded_rows),
            "operator_gated_entry_count": len(operator_gated_rows),
            "smoke_failure_count": len(smoke_failures),
            "runtime_smoke_failure_count": len(runtime_smoke_failures) + len(contract_dispatch_smoke_failures),
            "base_runtime_smoke_failure_count": len(runtime_smoke_failures),
            "contract_dispatch_smoke_failure_count": len(contract_dispatch_smoke_failures),
            "commands_hygiene_failure_count": len(hygiene_failures),
            "contract_hash_mismatch_count": 1 if contract_hash_mismatch else 0,
            "operator_gated_probe_count": len(probed_operator_rows),
            "unprobed_operator_gated_count": len(unprobed_operator_gated_subcommands),
        },
        "unprobed_operator_gated_subcommands": unprobed_operator_gated_subcommands,
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
