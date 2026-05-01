from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = PROJECT_ROOT / "scripts" / "ops"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import command_validity_bot as validity_src
from scripts.ops import infrastructure_autofix_bot as infra_src


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_command_validity_bot_accepts_repo_commands_surface() -> None:
    payload = validity_src.build_payload(PROJECT_ROOT, apply=False, timeout_sec=60)

    assert payload["overall_status"] in {"ready", "degraded"}
    stop_row = next(row for row in payload["command_rows"] if row["title"] == "Stop the stack")
    assert stop_row["validation_status"] == "operator_gated"
    assert payload["metrics"]["blocked_entry_count"] == 0
    assert payload["metrics"]["contract_hash_mismatch_count"] == 0
    assert payload["metrics"]["contract_dispatch_smoke_failure_count"] == 0
    assert payload["metrics"]["unprobed_operator_gated_count"] == 0
    assert payload["unprobed_operator_gated_subcommands"] == []


def test_opsctl_mutating_routes_keep_safe_probe_paths() -> None:
    probes = [
        ("feed-refresh", "--dry-run", "feed_refresh_dry_run=1"),
        ("livefeed-refresh", "--dry-run", "feed_refresh_dry_run=1"),
        ("storage-switch-local", "--dry-run", "storage_switch_dry_run=1"),
        ("storage-switch-external", "--dry-run", "storage_switch_dry_run=1"),
        ("storage-safe-eject", "--dry-run", "storage_switch_dry_run=1"),
        ("macro-auto-start", "--dry-run", "macro_auto_start_dry_run=1"),
        ("showcase-refresh", "--help", "Usage: opsctl.sh showcase-refresh"),
        ("system-explainers", "--help", "Usage: opsctl.sh system-explainers"),
    ]

    for subcommand, probe_arg, expected in probes:
        proc = subprocess.run(
            ["zsh", str(PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"), subcommand, probe_arg],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert proc.returncode == 0, (subcommand, proc.stdout, proc.stderr)
        assert expected in proc.stdout


def test_command_validity_bot_flags_missing_opsctl_alias(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write(
        project_root / "COMMANDS.md",
        """# Commands (Canonical)

## Most Used

### Broken command
```bash
cd /tmp
./scripts/ops/opsctl.sh missing-cmd --json
```
""",
    )
    _write(
        project_root / "scripts" / "ops" / "opsctl.sh",
        """#!/bin/zsh
cmd="${1:-help}"
case "$cmd" in
  known)
    echo ok
    ;;
  help|*)
    cat <<'EOF'
Usage: opsctl.sh <command>
  known [--json]
EOF
    ;;
esac
""",
    )
    _write(project_root / "scripts" / "runbook.sh", "#!/bin/zsh\necho ok\n")

    payload = validity_src.build_payload(project_root, apply=False, timeout_sec=15)

    assert payload["overall_status"] == "blocked"
    assert payload["metrics"]["blocked_entry_count"] == 1
    row = payload["command_rows"][0]
    assert "opsctl_dispatch_missing:missing-cmd" in row["issues"]
    assert "opsctl_help_missing:missing-cmd" in row["issues"]


def test_command_validity_bot_flags_missing_open_target(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    missing_pdf = project_root / "exports" / "reports" / "missing.pdf"
    _write(
        project_root / "COMMANDS.md",
        f"""# Commands (Canonical)

## Reports

### Broken open
```bash
cd {project_root}
open {missing_pdf}
```
""",
    )
    _write(
        project_root / "scripts" / "ops" / "opsctl.sh",
        """#!/bin/zsh
cmd="${1:-help}"
case "$cmd" in
  help|*)
    echo 'Usage: opsctl.sh <command>'
    ;;
esac
""",
    )
    _write(project_root / "scripts" / "runbook.sh", "#!/bin/zsh\necho ok\n")

    payload = validity_src.build_payload(project_root, apply=False, timeout_sec=15)

    assert payload["overall_status"] == "blocked"
    assert payload["metrics"]["blocked_entry_count"] == 1
    assert f"open_target_missing:{missing_pdf}" in payload["command_rows"][0]["issues"]


def test_command_validity_bot_blocks_failed_operator_route_probe(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write(
        project_root / "COMMANDS.md",
        """# Commands (Canonical)

## Schwab Auth

### Interactive Schwab authorization re-consent
```bash
cd /tmp
./scripts/ops/opsctl.sh token-refresh-interactive
```
""",
    )
    _write(
        project_root / "scripts" / "ops" / "opsctl.sh",
        """#!/bin/zsh
cmd="${1:-help}"
case "$cmd" in
  token-refresh-interactive)
    echo "ModuleNotFoundError: No module named 'scripts'" >&2
    exit 1
    ;;
  help|*)
    cat <<'EOF'
Usage: opsctl.sh <command>
  token-refresh-interactive [--json]
EOF
    ;;
esac
""",
    )
    _write(project_root / "scripts" / "runbook.sh", "#!/bin/zsh\necho ok\n")

    payload = validity_src.build_payload(project_root, apply=False, timeout_sec=15)

    assert payload["overall_status"] == "blocked"
    assert payload["metrics"]["blocked_entry_count"] == 0
    assert payload["metrics"]["contract_dispatch_smoke_failure_count"] == 1
    failed = [row for row in payload["contract_dispatch_smoke"] if not row["ok"]]
    assert failed[0]["subcommand"] == "token-refresh-interactive"
    assert "ModuleNotFoundError" in failed[0]["stderr_tail"]


def test_command_validity_bot_gates_mutating_operator_commands() -> None:
    assert validity_src._line_safety("./scripts/ops/opsctl.sh livefeed-refresh") == "operator_gated"
    assert validity_src._line_safety("./scripts/ops/opsctl.sh token-refresh --always-auth") == "operator_gated"
    assert validity_src._line_safety("./scripts/ops/opsctl.sh feed --source all --heavy") == "operator_gated"


def test_command_validity_bot_accepts_bare_help_alias_lines() -> None:
    aliases = validity_src._opsctl_help_aliases(
        """opsctl commands:
  stop
  status
  command-validity [--json]
"""
    )

    assert {"stop", "status", "command-validity"} <= aliases


def test_command_validity_bot_ignores_multiline_shell_fragments(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write(
        project_root / "COMMANDS.md",
        """# Commands (Canonical)

## Most Used

### Preview logs
```bash
find /tmp \\
  -type f -print0 \\
  | while IFS= read -r -d '' path; do
      [ -e "$path" ] && printf '%s\\n' "$path"
    done | sort
```
""",
    )
    _write(
        project_root / "scripts" / "ops" / "opsctl.sh",
        """#!/bin/zsh
cmd="${1:-help}"
case "$cmd" in
  help|*)
    cat <<'EOF'
Usage: opsctl.sh <command>
  status
EOF
    ;;
esac
""",
    )
    _write(project_root / "scripts" / "runbook.sh", "#!/bin/zsh\necho ok\n")

    payload = validity_src.build_payload(project_root, apply=False, timeout_sec=15)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["blocked_entry_count"] == 0


def test_command_validity_bot_writes_status_under_requested_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write(
        project_root / "COMMANDS.md",
        """# Commands (Canonical)

## Most Used

### Status
```bash
cd /tmp
./scripts/ops/opsctl.sh status
```
""",
    )
    _write(
        project_root / "scripts" / "ops" / "opsctl.sh",
        """#!/bin/zsh
cmd="${1:-help}"
case "$cmd" in
  status)
    echo ok
    ;;
  help|*)
    cat <<'EOF'
Usage: opsctl.sh <command>
  status
EOF
    ;;
esac
""",
    )
    _write(
        project_root / "scripts" / "ops" / "commands_hygiene_bot.py",
        """#!/usr/bin/env python3
import json
print(json.dumps({"overall_status": "ready", "commands_changed": False, "runbook_changed": False, "metrics": {"duplicate_entry_count": 0}}))
""",
    )
    _write(project_root / "scripts" / "runbook.sh", "#!/bin/zsh\necho ok\n")

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ops" / "command_validity_bot.py"),
            "--project-root",
            str(project_root),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    out_path = project_root / "governance" / "health" / "command_validity_latest.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["commands_path"] == str(project_root / "COMMANDS.md")
    assert payload["metrics"]["blocked_entry_count"] == 0


def test_infrastructure_autofix_bot_surfaces_command_validity_plan(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write(
        project_root / "COMMANDS.md",
        """# Commands (Canonical)

## Most Used

### Broken command
```bash
cd /tmp
./scripts/ops/opsctl.sh missing-cmd --json
```
""",
    )
    _write(
        project_root / "scripts" / "ops" / "opsctl.sh",
        """#!/bin/zsh
cmd="${1:-help}"
case "$cmd" in
  known)
    echo ok
    ;;
  help|*)
    cat <<'EOF'
Usage: opsctl.sh <command>
  known [--json]
EOF
    ;;
esac
""",
    )
    _write(project_root / "scripts" / "runbook.sh", "#!/bin/zsh\necho ok\n")

    payload = infra_src.build_payload(project_root, apply=False, timeout_sec=60)

    names = [row["name"] for row in payload["repair_plan"]]
    assert "command_validity" in names
