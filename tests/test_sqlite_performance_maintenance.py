import scripts.sqlite_performance_maintenance as maint


def test_checkpoint_mode_for_wal_uses_truncate_for_small_wal() -> None:
    assert maint._checkpoint_mode_for_wal(1.5, "auto", 8.0) == "truncate"


def test_checkpoint_mode_for_wal_uses_passive_for_large_wal() -> None:
    assert maint._checkpoint_mode_for_wal(12.0, "auto", 8.0) == "passive"


def test_checkpoint_mode_for_wal_respects_explicit_mode() -> None:
    assert maint._checkpoint_mode_for_wal(12.0, "restart", 8.0) == "restart"
