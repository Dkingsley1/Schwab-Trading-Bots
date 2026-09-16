from pathlib import Path
from types import SimpleNamespace

from scripts.ops import mlx_library_upgrade as src


def test_build_payload_uses_only_mlx_lock_rows(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(
        "\n".join(
            [
                "mlx==0.31.1",
                "mlx-lm==0.31.2",
                "numpy==2.4.0",
                "mlx-embedding-models==0.0.11",
                "parakeet-mlx==0.5.1",
            ]
        ),
        encoding="utf-8",
    )

    payload = src.build_payload(lock_path=lock, python_bin=Path("/venv/bin/python"))

    assert payload["ok"] is True
    assert payload["install_command"] == [
        "/venv/bin/python",
        "-m",
        "pip",
        "install",
        "-U",
        "mlx==0.31.1",
        "mlx-embedding-models==0.0.11",
        "mlx-lm==0.31.2",
        "parakeet-mlx==0.5.1",
    ]
    assert {row["package"] for row in payload["packages"]} == {
        "mlx",
        "mlx-lm",
        "mlx-embedding-models",
        "parakeet-mlx",
    }


def test_build_payload_all_scope_uses_exact_lock_file(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text("mlx==0.32.2\nriver==0.26.1\n", encoding="utf-8")

    payload = src.build_payload(
        lock_path=lock,
        python_bin=Path("/venv/bin/python"),
        scope="all",
    )

    assert payload["scope"] == "all"
    assert payload["install_command"] == [
        "/venv/bin/python",
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
        "-r",
        str(lock),
    ]
    assert {row["package"] for row in payload["packages"]} == {"mlx", "river"}


def test_virtualenv_python_path_is_not_resolved_to_base_interpreter(
    tmp_path: Path,
) -> None:
    base_python = tmp_path / "python3.14"
    base_python.touch()
    venv_python = tmp_path / ".venv314" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    selected = src._absolute_without_resolving_symlinks(venv_python)

    assert selected == venv_python
    assert selected != base_python


def test_transaction_preflight_fails_closed_without_maintenance_fences(
    tmp_path: Path, monkeypatch
) -> None:
    python_bin = tmp_path / "python"
    python_bin.touch()
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text("mlx==0.32.2\n", encoding="utf-8")
    monkeypatch.setattr(
        src,
        "maintenance_hold_snapshot",
        lambda _root: {"active": False, "valid": True},
    )
    monkeypatch.setattr(
        src, "maintenance_hold_token_authorized", lambda _hold, token="": False
    )

    preflight = src.transaction_preflight(
        project_root=tmp_path,
        python_bin=python_bin,
        lock_path=lock,
        runtime_processes=[],
    )

    assert preflight["ok"] is False
    assert set(preflight["failures"]) >= {
        "maintenance_acknowledgement_required",
        "active_maintenance_hold_required",
        "maintenance_token_required",
        "stack_stopped_flag_required",
    }


def test_transaction_preflight_accepts_stopped_authorized_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    python_bin = tmp_path / "python"
    python_bin.touch()
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text("mlx==0.32.2\n", encoding="utf-8")
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True)
    (health / src.STACK_STOPPED_FLAG.name).write_text("stopped\n", encoding="utf-8")
    monkeypatch.setattr(
        src,
        "maintenance_hold_snapshot",
        lambda _root: {"active": True, "valid": True, "token": "secret"},
    )
    monkeypatch.setattr(
        src,
        "maintenance_hold_token_authorized",
        lambda _hold, token="": token == "secret",
    )

    preflight = src.transaction_preflight(
        project_root=tmp_path,
        python_bin=python_bin,
        lock_path=lock,
        maintenance_token="secret",
        acknowledge_maintenance=True,
        runtime_processes=[],
    )

    assert preflight["ok"] is True
    assert preflight["failures"] == []


def test_command_result_classifies_native_signal(monkeypatch) -> None:
    monkeypatch.setattr(
        src.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=-6,
            stdout="",
            stderr="Abort trap",
        ),
    )

    result = src._command_result("native_smoke", ["python", "-c", "pass"])

    assert result["ok"] is False
    assert result["termination_signal"] == "SIGABRT"


def test_snapshot_never_reinstalls_running_pip(tmp_path: Path, monkeypatch) -> None:
    observed: list[str] = []

    def fake_result(_name: str, command: list[str], **_kwargs):
        observed.extend(command)
        return {"ok": True, "stdout": "mlx==0.32.2\n", "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(src, "_command_result", fake_result)

    snapshot, result = src._snapshot_environment(
        Path("/venv/bin/python"), tmp_path / "backups"
    )

    assert snapshot is not None
    assert result["ok"] is True
    assert observed[-2:] == ["--exclude", "pip"]


def test_validation_commands_include_full_suite_and_quant_capability() -> None:
    commands = src._validation_commands(
        Path("/venv/bin/python"),
        Path("/repo/config/requirements.lock.txt"),
        full_test=True,
        quant_capability=True,
    )

    by_name = {name: command for name, command, _timeout in commands}
    assert set(by_name) == {
        "pip_check",
        "mlx_runtime_audit",
        "quant_capability_smoke",
        "pytest_validation",
    }
    assert by_name["pytest_validation"] == [
        "/venv/bin/python",
        "-m",
        "pytest",
        "-q",
    ]
