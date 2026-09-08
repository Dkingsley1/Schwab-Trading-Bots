from pathlib import Path

from scripts.ops import runtime_dependency_profiles as src


def _lock_file(tmp_path: Path) -> Path:
    lock_file = tmp_path / "requirements.lock.txt"
    lock_file.write_text(
        "\n".join(
            [
                "SQLAlchemy==2.0.48",
                "duckdb==1.5.0",
                "torch==2.10.0",
                "mlx-whisper==0.4.3",
                "fastapi==0.135.3",
                "requests==2.32.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return lock_file


def test_runtime_dependency_profiles_audit_does_not_mutate_profile_locks(
    tmp_path: Path,
) -> None:
    lock_file = _lock_file(tmp_path)
    profile_dir = tmp_path / "profiles"

    payload = src.build_payload(lock_file, profile_dir)

    assert payload["ok"] is False
    assert payload["overall_status"] == "degraded"
    assert payload["apply_requested"] is False
    assert payload["profile_drift_count"] == 4
    assert not profile_dir.exists()


def test_runtime_dependency_profiles_apply_builds_profile_locks(tmp_path: Path) -> None:
    lock_file = _lock_file(tmp_path)

    profile_dir = tmp_path / "profiles"
    payload = src.build_payload(lock_file, profile_dir, apply=True)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["apply_requested"] is True
    assert payload["applied_profile_count"] == 4
    assert payload["profile_drift_count"] == 0
    assert (profile_dir / "live.lock.txt").exists()
    assert (profile_dir / "research.lock.txt").exists()
    assert (profile_dir / "media.lock.txt").exists()
    assert (profile_dir / "ops.lock.txt").exists()
    assert payload["profile_counts"]["live"] >= 2
    assert payload["profile_counts"]["research"] >= 1
    assert payload["profile_counts"]["media"] >= 1
    assert payload["profile_counts"]["ops"] >= 1

    mtimes = {
        profile: (profile_dir / f"{profile}.lock.txt").stat().st_mtime_ns
        for profile in src.PROFILE_ORDER
    }
    second = src.build_payload(lock_file, profile_dir, apply=True)
    assert second["ok"] is True
    assert second["applied"] is False
    assert second["observed_profile_drift_count"] == 0
    assert mtimes == {
        profile: (profile_dir / f"{profile}.lock.txt").stat().st_mtime_ns
        for profile in src.PROFILE_ORDER
    }
