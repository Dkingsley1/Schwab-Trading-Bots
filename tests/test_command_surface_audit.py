from pathlib import Path
import json
import sys

import pytest

from scripts.ops import command_surface_audit as audit


def fixture(tmp_path, code="./scripts/ops/opsctl.sh example --apply"):
    scripts = tmp_path / "scripts" / "ops"
    scripts.mkdir(parents=True)
    (tmp_path / "COMMANDS.md").write_text(
        f"# Commands\n\n## Most Used\n\n### Example\n```bash\n{code}\n```\n"
    )
    (scripts / "opsctl.sh").write_text(
        '#!/bin/zsh\ncase "$1" in\n  example)\n'
        '    exec "$PY" "$PROJECT_ROOT/scripts/ops/example.py" "$@"\n    ;;\nesac\n'
    )
    (scripts / "example.py").write_text("raise RuntimeError('must never execute')\n")
    return scripts


def test_all_source_checks_are_nonexecuting_and_not_functional_proof(
    tmp_path, monkeypatch
):
    fixture(tmp_path)
    calls = []

    def syntax_only(cmd, **kwargs):
        assert cmd[:3] == ["/bin/zsh", "-f", "-n"]
        calls.append(cmd)
        return {"rc": 0, "timed_out": False}

    monkeypatch.setattr(audit, "run_bounded_process_group", syntax_only)
    payload = audit.build_payload(tmp_path)
    assert payload["overall_status"] == "ready"
    assert len(calls) == 1
    row = payload["command_rows"][0]
    assert row["implementation_paths"] == [
        "scripts/ops/example.py",
        "scripts/ops/opsctl.sh",
    ]
    assert row["validation_status"] == "static_pass"
    assert not row["execution_authorized"] and not row["exact_arguments_verified"]
    assert payload["metrics"]["functionally_verified_entry_count"] == 0
    assert not any(payload["authority"].values())


@pytest.mark.parametrize("failure", ["missing", "syntax"])
def test_inspects_dispatch_target_not_only_opsctl_wrapper(tmp_path, failure):
    scripts = fixture(tmp_path)
    target = scripts / "example.py"
    if failure == "missing":
        target.unlink()
    else:
        target.write_text("def broken(:\n")
    payload = audit.build_payload(tmp_path)
    assert payload["overall_status"] == "blocked"
    assert "scripts/ops/example.py" in payload["command_rows"][0]["issues"][0]


def test_source_symlink_is_rejected_without_reading_target(tmp_path):
    scripts = fixture(tmp_path)
    target = scripts / "example.py"
    target.unlink()
    target.symlink_to(tmp_path / "not-to-be-read")
    payload = audit.build_payload(tmp_path)
    assert payload["overall_status"] == "blocked"
    assert "symlink_source_not_inspected" in str(payload["command_rows"][0]["issues"])


def test_duplicate_snippets_flagged_without_deleting_entries(tmp_path):
    fixture(tmp_path)
    commands = tmp_path / "COMMANDS.md"
    commands.write_text(
        commands.read_text()
        + "\n### Same action\n```bash\n./scripts/ops/opsctl.sh example --apply\n```\n"
    )
    before = commands.read_bytes()
    payload = audit.build_payload(tmp_path)
    assert payload["metrics"]["entry_count"] == 2
    assert payload["metrics"]["duplicate_group_count"] == 1
    assert commands.read_bytes() == before
    assert payload["metrics"]["unique_source_count"] == 2


def test_missing_alias_and_shell_only_entries_are_not_certified(tmp_path):
    fixture(tmp_path, "./scripts/ops/opsctl.sh absent")
    payload = audit.build_payload(tmp_path)
    assert payload["command_rows"][0]["issues"] == ["opsctl_dispatch_missing:absent"]
    (tmp_path / "COMMANDS.md").write_text(
        "# Commands\n\n## Tools\n\n### Delete\n```bash\nrm -rf /never-execute\n```\n"
    )
    payload = audit.build_payload(tmp_path)
    assert payload["command_rows"][0]["validation_status"] == "unverified"
    assert payload["metrics"]["functionally_verified_entry_count"] == 0


def test_budget_deferral_is_not_success(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(audit, "MAX_TOTAL_BYTES", 0)
    payload = audit.build_payload(tmp_path)
    assert payload["overall_status"] == "degraded"
    assert payload["metrics"]["unverified_entry_count"] == 1


def test_env_prefixed_multiline_route_is_audited(tmp_path):
    fixture(
        tmp_path, "EXAMPLE=1 ./scripts/ops/opsctl.sh " + "\\" + "\n example --apply"
    )
    assert audit.build_payload(tmp_path)["command_rows"][0]["opsctl_subcommands"] == [
        "example"
    ]


def test_native_owner_uses_nonexecuting_mode():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/ops/run_command_validity_launchd.sh").read_text()
    assert "--safe-audit --timeout-sec 30 --summary-json" in text


def test_cli_keeps_compact_health_and_does_not_rewrite_unchanged_detail(
    tmp_path, monkeypatch, capsys
):
    from scripts.ops import command_validity_bot as cli

    fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "--project-root", str(tmp_path), "--safe-audit", "--summary-json"],
    )
    monkeypatch.setattr(
        cli,
        "build_payload",
        lambda *a, **k: pytest.fail("legacy executable probes must not run"),
    )
    assert cli.main() == 0
    first = json.loads(capsys.readouterr().out)
    detail = Path(first["surface_report"])
    before = detail.stat().st_mtime_ns
    assert cli.main() == 0
    second = json.loads(capsys.readouterr().out)
    assert detail.stat().st_mtime_ns == before
    assert second["timestamp_utc"] != first["timestamp_utc"]
    health = json.loads(
        (tmp_path / "governance/health/command_validity_latest.json").read_text()
    )
    assert "command_rows" not in health
    assert len(json.loads(detail.read_text())["command_rows"]) == 1


def test_safe_mode_cannot_apply_source_changes(tmp_path, monkeypatch):
    from scripts.ops import command_validity_bot as cli

    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "--project-root", str(tmp_path), "--safe-audit", "--apply"],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2
    assert not (tmp_path / "governance").exists()


def test_protected_project_root_rejected_before_source_read(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audit, "inspect_storage_path", lambda _: {"status": "protected"}
    )
    with pytest.raises(ValueError, match="protected_or_unavailable"):
        audit._read_local(tmp_path, "COMMANDS.md")
