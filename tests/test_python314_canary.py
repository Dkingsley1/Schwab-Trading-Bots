import base64
import json

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


def test_installer_artifact_step_checks_hash_and_sigstore_digest(tmp_path) -> None:
    installer = tmp_path / "python-3.14.5-macos11.pkg"
    installer.write_bytes(b"x" * 50_000_001)
    digest = src._sha256_file(installer)
    canonical_body = {
        "apiVersion": "0.0.1",
        "kind": "hashedrekord",
        "spec": {"data": {"hash": {"algorithm": "sha256", "value": digest}}},
    }
    sigstore = tmp_path / "python-3.14.5-macos11.pkg.sigstore"
    sigstore.write_text(
        json.dumps(
            {
                "verificationMaterial": {
                    "tlogEntries": [
                        {
                            "canonicalizedBody": base64.b64encode(
                                json.dumps(canonical_body).encode("utf-8")
                            ).decode("ascii")
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    step = src._installer_artifact_step(installer, sigstore, digest)

    assert step["ok"] is True
    assert step["sha256_ok"] is True
    assert step["sigstore_ok"] is True


def test_transition_readiness_keeps_production_flip_blocked() -> None:
    steps = [
        {"name": "python3145_download_artifact", "ok": True},
        {"name": "homebrew_python314_exact_version", "ok": True},
        {"name": "venv_python314_exact_version", "ok": True},
        {"name": "production_anchor_python312", "ok": True},
        {
            "name": "critical_runtime_packages",
            "ok": False,
            "missing_packages": ["mlx"],
        },
    ]

    readiness = src._transition_readiness(
        steps=steps,
        import_steps=[],
        smoke_steps=[],
        signature_step={"ok": False},
        bootstrap_ok=False,
        smoke_ok=False,
    )

    assert readiness["production_runtime_change_allowed"] is False
    assert readiness["promotion_allowed"] is False
    assert "critical_runtime_packages_missing:mlx" in readiness["blockers"]
    assert "bootstrap_not_green" in readiness["blockers"]


def test_transition_readiness_allows_approved_runtime_flip() -> None:
    steps = [
        {"name": "python3145_download_artifact", "ok": True},
        {"name": "homebrew_python314_exact_version", "ok": True},
        {"name": "venv_python314_exact_version", "ok": True},
        {"name": "production_anchor_python312", "ok": True},
        {
            "name": "lock_alignment",
            "ok": False,
            "missing_count": 4,
            "mismatch_count": 63,
        },
        {
            "name": "py314_compatibility_alignment",
            "ok": True,
            "exempt_missing_packages": ["mlx-cluster", "mlx-data", "mlx-graphs", "pandas-ta"],
            "version_mismatch_count": 63,
        },
        {"name": "critical_runtime_packages", "ok": True, "missing_packages": []},
        {"name": "test_tooling_packages", "ok": True, "missing_packages": []},
    ]

    readiness = src._transition_readiness(
        steps=steps,
        import_steps=[],
        smoke_steps=[{"name": "session_ready_check", "ok": True}],
        signature_step={"ok": True},
        bootstrap_ok=True,
        smoke_ok=True,
        runtime_flip_approved=True,
    )

    assert readiness["production_runtime_change_allowed"] is True
    assert readiness["runtime_flip_approved"] is True
    assert readiness["current_transition_state"] == "runtime_flip_approved"
    assert readiness["warnings"] == []
    assert readiness["compatibility_notes"] == [
        "strict_312_lock_alignment_documented_for_py314:exempt_missing=4,version_mismatches_allowed=63"
    ]


def test_py314_compatibility_alignment_exempts_known_satellites() -> None:
    alignment = {
        "missing_packages": ["mlx-audio", "pandas-ta"],
        "version_mismatches": [
            {
                "package": "numpy",
                "lock_version": "2.2.6",
                "installed_version": "2.4.2",
            }
        ],
    }

    step = src._py314_compatibility_alignment_step(
        alignment,
        ("mlx-audio", "pandas-ta"),
    )

    assert step["ok"] is True
    assert step["blocking_missing_packages"] == []
    assert step["exempt_missing_packages"] == ["mlx-audio", "pandas-ta"]
    assert step["version_mismatch_count"] == 1


def test_py314_compatibility_alignment_blocks_unexpected_missing() -> None:
    alignment = {
        "missing_packages": ["duckdb", "mlx-audio"],
        "version_mismatches": [],
    }

    step = src._py314_compatibility_alignment_step(alignment, ("mlx-audio",))

    assert step["ok"] is False
    assert step["blocking_missing_packages"] == ["duckdb"]
