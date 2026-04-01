import scripts.run_shadow_training_loop as loop


def test_auto_retrain_memory_guard_bypasses_high_swap_when_free_memory_is_strong(monkeypatch) -> None:
    monkeypatch.setattr(
        loop,
        "_memory_guard_snapshot",
        lambda: {"free_pct": 90.0, "swap_used_gb": 21.5},
    )

    ok, snapshot, reason = loop._auto_retrain_memory_ok(
        min_free_pct=20.0,
        max_swap_gb=2.2,
        high_free_pct_swap_bypass=85.0,
        soft_max_swap_gb=24.0,
    )

    assert ok is True
    assert snapshot["swap_guard_bypassed"] == 1.0
    assert snapshot["swap_guard_soft_max_gb"] == 24.0
    assert "swap_soft_bypass" in reason


def test_auto_retrain_memory_guard_still_blocks_when_soft_swap_limit_is_exceeded(monkeypatch) -> None:
    monkeypatch.setattr(
        loop,
        "_memory_guard_snapshot",
        lambda: {"free_pct": 92.0, "swap_used_gb": 26.0},
    )

    ok, snapshot, reason = loop._auto_retrain_memory_ok(
        min_free_pct=20.0,
        max_swap_gb=2.2,
        high_free_pct_swap_bypass=85.0,
        soft_max_swap_gb=24.0,
    )

    assert ok is False
    assert "swap_guard_bypassed" not in snapshot
    assert "swap_above_threshold" in reason


def test_auto_retrain_memory_guard_uses_available_pct_as_headroom(monkeypatch) -> None:
    monkeypatch.setattr(
        loop,
        "_memory_guard_snapshot",
        lambda: {"free_pct": 12.0, "available_pct": 58.0, "swap_used_gb": 10.0},
    )

    ok, snapshot, reason = loop._auto_retrain_memory_ok(
        min_free_pct=20.0,
        max_swap_gb=2.2,
        high_free_pct_swap_bypass=55.0,
        soft_max_swap_gb=12.0,
    )

    assert ok is True
    assert snapshot["headroom_pct"] == 58.0
    assert snapshot["swap_guard_bypassed"] == 1.0
    assert "headroom_pct=58.0" in reason
