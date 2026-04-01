from scripts.ops import python314_canary as src


def test_package_alignment_flags_missing_and_mismatched_packages() -> None:
    lock_versions = {
        "mlx": "0.30.6",
        "numpy": "2.2.6",
        "pandas": "3.0.1",
    }
    installed_versions = {
        "numpy": "2.4.2",
        "pandas": "3.0.1",
        "pytest": "9.0.2",
    }

    alignment = src._package_alignment(lock_versions, installed_versions)

    assert alignment["ok"] is False
    assert alignment["missing_packages"] == ["mlx"]
    assert alignment["extra_packages"] == ["pytest"]
    assert alignment["version_mismatches"] == [
        {
            "package": "numpy",
            "lock_version": "2.2.6",
            "installed_version": "2.4.2",
        }
    ]


def test_required_packages_step_fails_when_runtime_packages_missing() -> None:
    installed_versions = {
        "numpy": "2.2.6",
        "pandas": "3.0.1",
    }

    step = src._required_packages_step(
        "critical_runtime_packages",
        installed_versions,
        ("mlx", "mlx-metal", "mlx-lm"),
    )

    assert step["ok"] is False
    assert step["missing_packages"] == ["mlx", "mlx-lm", "mlx-metal"]
    assert step["required_packages"] == ["mlx", "mlx-lm", "mlx-metal"]


def test_import_step_marks_module_not_found_as_failure(monkeypatch) -> None:
    def fake_run(cmd: list[str]) -> tuple[int, str, str]:
        return 1, "", "ModuleNotFoundError: No module named 'mlx'"

    monkeypatch.setattr(src, "_run", fake_run)

    step = src._import_step("mlx_core_import", src.DEFAULT_VENV / "bin" / "python", "import mlx.core as mx")

    assert step["ok"] is False
    assert step["stderr_tail"] == "ModuleNotFoundError: No module named 'mlx'"
