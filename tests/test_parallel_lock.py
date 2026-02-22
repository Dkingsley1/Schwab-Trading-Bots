from pathlib import Path

from scripts.run_parallel_shadows import _acquire_singleton_lock


def test_parallel_lock_is_singleton(tmp_path: Path) -> None:
    lock_path = tmp_path / "parallel_shadow.lock"

    h1 = _acquire_singleton_lock(lock_path)
    assert h1 is not None

    h2 = _acquire_singleton_lock(lock_path)
    assert h2 is None

    h1.close()
