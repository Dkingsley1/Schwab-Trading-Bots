from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from scripts.ops import overnight_training_window as src


def test_overnight_training_launch_uses_performance_core_env(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_ops(project_root: Path, args: list[str], *, timeout_seconds: int) -> dict[str, object]:
        if args[:1] == ["ingestion-storage-control"]:
            return {
                "ok": True,
                "returncode": 0,
                "json": {
                    "overall_status": "ready",
                    "backpressure": {
                        "total_pending_lines": 0,
                        "core_pending_lines": 0,
                        "oldest_pending_age_seconds": 0,
                    },
                },
            }
        if args[:1] == ["training-runtime-control"]:
            return {
                "ok": True,
                "returncode": 0,
                "json": {
                    "overall_status": "ready",
                    "training_launch_contract": {
                        "launch_allowed": True,
                        "mode": "canary_training_allowed",
                        "recommended_batch_size": 2,
                        "recommended_retrain_command": ["./scripts/ops/opsctl.sh", "retrain-force-targeted"],
                    },
                    "host_training_headroom_gate": {
                        "status": "ready",
                        "batch_cap": 2,
                        "memory_status": "clear",
                        "memory_decision": "safe_to_train",
                    },
                },
            }
        return {"ok": True, "returncode": 0, "json": {"overall_status": "ready"}}

    def fake_run_json(
        command: list[str],
        *,
        project_root: Path,
        timeout_seconds: int,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, object]:
        calls.append({"command": command, "env_overrides": env_overrides or {}})
        return {"ok": True, "returncode": 0, "json": {"overall_status": "ready"}}

    monkeypatch.setattr(src, "_run_ops", fake_run_ops)
    monkeypatch.setattr(src, "_run_json", fake_run_json)
    now = datetime.now().astimezone()
    record = src.run_cycle(
        project_root=tmp_path,
        apply=True,
        end_local=now + timedelta(hours=1),
        small_start=now + timedelta(minutes=10),
        small_end=now + timedelta(minutes=20),
        large_limit=20,
        small_limit=2,
        command_timeout_seconds=5,
    )

    assert record["launch_attempted"] is True
    assert calls
    env = calls[-1]["env_overrides"]
    assert env["BOT_CPU_ALLOCATION_POLICY"] == "performance_core_primary"
    assert env["BOT_CPU_QOS_POLICY"] == "performance_core_primary_no_background_writer"
    assert env["BOT_CPU_SCHEDULER_INTENT"] == "performance_core_training"
    assert env["TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN"] == "1"
    assert env["TRAINING_PCORE_MAX_WORKERS"] == "3"
    assert record["protected_volumes"]["VIDEO"] == "never_touched"


def test_overnight_training_uses_super_drainer_before_writer_cycle_when_storage_is_severe(tmp_path: Path, monkeypatch) -> None:
    ops_calls: list[list[str]] = []

    def fake_run_ops(project_root: Path, args: list[str], *, timeout_seconds: int) -> dict[str, object]:
        ops_calls.append(list(args))
        if args[:1] == ["ingestion-storage-control"] and not any(call[:1] == ["backpressure-super-drainer"] for call in ops_calls):
            return {
                "ok": True,
                "returncode": 0,
                "json": {
                    "overall_status": "blocked",
                    "severity": "high",
                    "pressure_index": 2.0,
                    "backpressure": {
                        "total_pending_lines": 250000,
                        "core_pending_lines": 240000,
                        "oldest_pending_age_seconds": 3600,
                        "pending_lines_threshold": 15000,
                        "oldest_age_threshold_seconds": 240,
                    },
                },
            }
        if args[:1] == ["ingestion-storage-control"]:
            return {
                "ok": True,
                "returncode": 0,
                "json": {
                    "overall_status": "ready",
                    "severity": "stable",
                    "backpressure": {
                        "total_pending_lines": 0,
                        "core_pending_lines": 0,
                        "oldest_pending_age_seconds": 0,
                    },
                },
            }
        if args[:1] == ["training-runtime-control"]:
            return {
                "ok": True,
                "returncode": 0,
                "json": {
                    "overall_status": "blocked",
                    "training_launch_contract": {
                        "launch_allowed": False,
                        "mode": "prep_only",
                        "launch_blockers": ["host_memory_relief_active"],
                        "recommended_command": [],
                    },
                    "host_training_headroom_gate": {"status": "blocked", "batch_cap": 0},
                },
            }
        return {"ok": True, "returncode": 0, "json": {"overall_status": "ready"}}

    monkeypatch.setattr(src, "_run_ops", fake_run_ops)
    now = datetime.now().astimezone()
    record = src.run_cycle(
        project_root=tmp_path,
        apply=True,
        end_local=now + timedelta(hours=1),
        small_start=now + timedelta(minutes=10),
        small_end=now + timedelta(minutes=20),
        large_limit=20,
        small_limit=2,
        command_timeout_seconds=5,
    )

    assert ["backpressure-super-drainer", "--apply", "--max-waves", "5", "--target-pending-lines", "15000", "--json"] in ops_calls
    assert any(call[:2] == ["writer-cycle-coordinator", "--apply"] for call in ops_calls)
    assert record["steps"]["backpressure_super_drainer"]["ok"] is True


def test_overnight_training_window_stops_at_target_across_guarded_waves(tmp_path: Path, monkeypatch) -> None:
    requested_limits: list[int] = []
    launched_commands: list[list[str]] = []

    def fake_run_ops(project_root: Path, args: list[str], *, timeout_seconds: int) -> dict[str, object]:
        if args[:1] == ["ingestion-storage-control"]:
            return {
                "ok": True,
                "returncode": 0,
                "json": {
                    "overall_status": "ready",
                    "backpressure": {
                        "total_pending_lines": 0,
                        "core_pending_lines": 0,
                        "oldest_pending_age_seconds": 0,
                    },
                },
            }
        if args[:1] == ["training-runtime-control"]:
            limit = int(args[args.index("--limit") + 1])
            requested_limits.append(limit)
            batch = min(limit, 30)
            bot_ids = ",".join(f"bot_{idx}" for idx in range(batch))
            return {
                "ok": True,
                "returncode": 0,
                "json": {
                    "overall_status": "ready",
                    "training_launch_contract": {
                        "launch_allowed": True,
                        "mode": "canary_training_allowed",
                        "recommended_batch_size": batch,
                        "recommended_retrain_command": [
                            "./scripts/ops/opsctl.sh",
                            "retrain-force-targeted",
                            "--include-bot-ids",
                            bot_ids,
                        ],
                    },
                    "host_training_headroom_gate": {
                        "status": "ready",
                        "batch_cap": batch,
                        "memory_status": "clear",
                        "memory_decision": "safe_to_train",
                    },
                },
            }
        return {"ok": True, "returncode": 0, "json": {"overall_status": "ready"}}

    def fake_run_json(
        command: list[str],
        *,
        project_root: Path,
        timeout_seconds: int,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, object]:
        launched_commands.append(command)
        return {"ok": True, "returncode": 0, "json": {"overall_status": "ready"}}

    monkeypatch.setattr(src, "_run_ops", fake_run_ops)
    monkeypatch.setattr(src, "_run_json", fake_run_json)
    monkeypatch.setattr(src.time, "sleep", lambda _: None)

    summary = src.run_window(
        project_root=tmp_path,
        apply=True,
        end_local_text="23:59",
        small_start_text="23:58",
        small_end_text="23:59",
        large_limit=30,
        small_limit=2,
        window_target=50,
        poll_seconds=5,
        command_timeout_seconds=5,
        max_cycles=10,
    )

    assert requested_limits == [30, 20]
    assert len(launched_commands) == 2
    assert summary["launched_batch_total"] == 50
    assert summary["remaining_target"] == 0
    assert summary["stop_reason"] == "window_target_reached"
