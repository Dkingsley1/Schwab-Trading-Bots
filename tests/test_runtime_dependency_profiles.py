from pathlib import Path

from scripts.ops import runtime_dependency_profiles as src


def test_runtime_dependency_profiles_builds_profile_locks(tmp_path: Path) -> None:
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

    profile_dir = tmp_path / "profiles"
    payload = src.build_payload(lock_file, profile_dir)

    assert payload["ok"] is True
    assert (profile_dir / "live.lock.txt").exists()
    assert (profile_dir / "research.lock.txt").exists()
    assert (profile_dir / "media.lock.txt").exists()
    assert (profile_dir / "ops.lock.txt").exists()
    assert payload["profile_counts"]["live"] >= 2
    assert payload["profile_counts"]["research"] >= 1
    assert payload["profile_counts"]["media"] >= 1
    assert payload["profile_counts"]["ops"] >= 1
