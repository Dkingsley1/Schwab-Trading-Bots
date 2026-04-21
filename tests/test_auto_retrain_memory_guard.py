import json
from pathlib import Path

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


def test_spawn_auto_retrain_labels_source_and_persists_log(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeProc:
        pid = 43210

    def _fake_popen(cmd, cwd=None, env=None, stdout=None, stderr=None):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["env"] = dict(env or {})
        captured["stdout_name"] = getattr(stdout, "name", "")
        captured["stderr"] = stderr
        return _FakeProc()

    monkeypatch.setattr(loop, "resolve_runtime_python", lambda project_root: Path("/tmp/fake-python"))
    monkeypatch.setattr(loop, "_shadow_profile_name", lambda: "aggressive")
    monkeypatch.setattr(loop.subprocess, "Popen", _fake_popen)

    proc = loop._spawn_auto_retrain(
        project_root=str(tmp_path),
        underperformers=3,
        sample_recommendations=[{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy"}],
        broker="schwab",
    )

    assert proc.pid == 43210
    env = captured["env"]
    assert env["RETRAIN_TRIGGER_SOURCE"] == "shadow_training_loop_auto_retrain"
    assert env["RETRAIN_TRIGGER_BROKER"] == "schwab"
    assert env["RETRAIN_TRIGGER_PROFILE"] == "aggressive"
    log_path = Path(env["RETRAIN_LAUNCH_LOG_PATH"])
    assert log_path.exists()
    assert log_path == Path(captured["stdout_name"])
    assert captured["stderr"] == loop.subprocess.STDOUT

    event_path = Path(loop._auto_retrain_log_path(str(tmp_path), broker="schwab"))
    loop.JsonlWriteBuffer.shared().flush_all()
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert events[-1]["trigger_source"] == "shadow_training_loop_auto_retrain"
    assert events[-1]["log_path"] == str(log_path)
