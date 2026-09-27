import os
from datetime import datetime, timedelta, timezone

from core.stack_restart_coordination import (
    engage_stack_restart_fence,
    release_stack_restart_fence,
    stack_restart_fence_snapshot,
)


def test_restart_fence_is_exclusive_and_token_owned(tmp_path) -> None:
    first = engage_stack_restart_fence(tmp_path, owner_pid=os.getpid(), ttl_seconds=120)
    assert first["acquired"] is True
    assert first["active"] is True

    second = engage_stack_restart_fence(tmp_path, owner_pid=os.getpid(), ttl_seconds=120)
    assert second["acquired"] is False
    assert second["acquire_error"] == "stack_restart_already_in_progress"

    denied = release_stack_restart_fence(tmp_path, expected_token="wrong")
    assert denied["released"] is False
    assert stack_restart_fence_snapshot(tmp_path)["active"] is True

    released = release_stack_restart_fence(tmp_path, expected_token=str(first["token"]))
    assert released["released"] is True
    assert stack_restart_fence_snapshot(tmp_path)["active"] is False


def test_dead_owner_fence_is_recoverable(tmp_path) -> None:
    first = engage_stack_restart_fence(tmp_path, owner_pid=987654321, ttl_seconds=120)
    assert first["acquired"] is True
    assert first["active"] is False
    assert first["reason"] == "stack_restart_owner_not_alive"

    replacement = engage_stack_restart_fence(tmp_path, owner_pid=os.getpid(), ttl_seconds=120)
    assert replacement["acquired"] is True
    assert replacement["active"] is True


def test_expired_fence_is_not_active(tmp_path) -> None:
    first = engage_stack_restart_fence(tmp_path, owner_pid=os.getpid(), ttl_seconds=60)
    engaged = datetime.fromisoformat(str(first["engaged_at_utc"]))
    later = engaged + timedelta(seconds=61)

    snapshot = stack_restart_fence_snapshot(tmp_path, now_utc=later.astimezone(timezone.utc))
    assert snapshot["active"] is False
    assert snapshot["expired"] is True
    assert snapshot["reason"] == "stack_restart_fence_expired"
