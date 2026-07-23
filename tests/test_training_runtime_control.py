import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import training_runtime_control as src
from scripts.ops import runtime_snapshot_cache_control as snapshot_cache_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _fresh_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_runtime_snapshot_cache_accepts_intrinsically_ready_snapshot(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 200, "sequence_count": 8})
    _write_json(health / "training_runtime_control_latest.json", {"snapshot_ready": False, "precompute_targets": []})
    _write_json(health / "retrain_artifact_freshness_latest.json", {"ok": True})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0})

    payload = snapshot_cache_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["cache_health"]["snapshot_ready"] is True
    assert payload["cache_health"]["snapshot_ready_source"] == "snapshot_payload"


def test_training_runtime_control_surfaces_runtime_backend_parity(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 10, "sequence_count": 2})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 88.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": ["python", "weekly_retrain.py"]}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {}})
    _write_json(walk / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0, "seed_queue": []})

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["runtime_backend_parity"]["parity_state"] == "ready"
    assert payload["runtime_backend_parity"]["native_contract"]["runtime_training_supported"] is True
    assert payload["training_launch_contract"]["mode"] == "blocked"


def test_training_runtime_control_blocks_on_missing_mlx_failure(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 10, "sequence_count": 2})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 88.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": ["python", "weekly_retrain.py"]}})
    _write_json(
        health / "training_success_latest.json",
        {"confirmed_training_success": False, "failure_details": [{"reason": "ModuleNotFoundError: No module named 'mlx'"}]},
    )
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {}})
    _write_json(walk / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0, "seed_queue": []})

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": False, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["runtime_backend_parity"]["mlx_failure_detected"] is True
    assert "install or repair MLX" in " ".join(payload["recommended_actions"])


def test_training_runtime_control_degrades_when_runtime_is_ready_but_repair_work_remains(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 10, "sequence_count": 2})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked", "training_quality_score": 12.0, "top_priorities": ["promotion_coverage"], "targeted_actions": {"targeted_retrain_bot_ids": ["bot_a"]}})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": ["python", "weekly_retrain.py"]}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": False, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {}})
    _write_json(walk / "coverage_seed_latest.json", {"coverage_shortfall_bots": 2, "seed_queue": [{"bot_id": "bot_a"}]})

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["coverage_repair_ready"] is True
    assert payload["training_launch_contract"]["mode"] == "prep_only"
    assert payload["training_launch_contract"]["launch_allowed"] is False
    assert payload["training_launch_contract"]["prep_allowed"] is True


def test_training_runtime_control_allows_canary_batch_when_gates_clear(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 1,
            "seed_queue": [
                {"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0},
                {"bot_id": "brain_refinery_v35_dmi_state_machine", "current_runs": 0, "runs_remaining": 12, "priority": 39.0, "needs_runtime_input_repair": True, "actions": ["repair_runtime_inputs"]},
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    contract = payload["training_launch_contract"]
    assert contract["mode"] == "canary_training_allowed"
    assert contract["launch_allowed"] is True
    assert contract["canary_batch"][0]["bot_id"] == "brain_refinery_v10_seasonal"
    assert contract["repair_first_targets"][0]["bot_id"] == "brain_refinery_v35_dmi_state_machine"
    assert "brain_refinery_v10_seasonal" in contract["recommended_retrain_command"]


def test_training_runtime_control_keeps_full_canary_when_writer_active_inside_buffer(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.02,
            "backpressure": {
                "total_pending_lines": 625,
                "oldest_pending_age_seconds": 40.0,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "writer_state_before": {
                "active": True,
                "running": True,
                "status": "running",
                "current_step": "shard_linking",
                "completed_shard_count": 11,
                "planned_shard_count": 14,
                "progress_age_minutes": 0.5,
                "cycle_age_minutes": 2.0,
            },
        },
    )
    _write_json(health / "backlog_pcore_accelerator_latest.json", {"overall_status": "ready", "bulletproof_score": {"score": 95, "letter": "A"}})
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 4,
            "seed_queue": [
                {"bot_id": f"brain_refinery_v{i}", "current_runs": 10, "runs_remaining": 0, "priority": 40.0 - i}
                for i in range(10, 15)
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is True
    assert contract["recommended_batch_size"] == 4
    assert contract["pretraining_drain_buffer"]["status"] == "writer_active_backlog_green"
    assert contract["pretraining_drain_buffer"]["safe_to_launch_now"] is True


def test_training_runtime_control_blocks_training_when_writer_progress_stale(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.02,
            "backpressure": {
                "total_pending_lines": 625,
                "oldest_pending_age_seconds": 40.0,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "writer_state_before": {
                "active": True,
                "running": True,
                "status": "running",
                "current_step": "shard_linking",
                "completed_shard_count": 11,
                "planned_shard_count": 14,
                "progress_age_minutes": 16.0,
                "cycle_age_minutes": 20.0,
            },
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 1, "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0}]},
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is False
    assert contract["pretraining_drain_buffer"]["status"] == "writer_attention_required"
    assert "writer_progress_stale_before_training" in contract["launch_blockers"]
    assert contract["recommended_prep_commands"][0][-3:] == ["--apply", "--skip-maintenance", "--json"]


def test_training_runtime_control_recommends_fast_handoff_for_completed_writer_lock(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(
        health / "health_gates_latest.json",
        {
            "recommended_operating_mode": "normal",
            "inputs": {
                "backpressure_overload_severe": True,
                "backpressure_pending_lines": 42000,
                "backpressure_oldest_pending_age_seconds": 900.0,
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 1.2,
            "backpressure": {
                "total_pending_lines": 42000,
                "oldest_pending_age_seconds": 900.0,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"overload": True, "pending_lines_total": 42000, "oldest_pending_age_seconds": 900.0},
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "writer_state_before": {
                "active": True,
                "active_source": "completed_lock_handoff_needed",
                "running": False,
                "status": "ok",
                "current_step": "complete",
                "writer_lock_held": True,
                "complete_lock_handoff_needed": True,
                "child_writer_active": False,
                "completed_shard_count": 14,
                "planned_shard_count": 14,
                "progress_age_minutes": 0.5,
                "cycle_age_minutes": 12.0,
            },
            "summary": {"completed_writer_lock_handoff_needed": True},
        },
    )
    _write_json(health / "backlog_pcore_accelerator_latest.json", {"overall_status": "ready", "bulletproof_score": {"score": 95, "letter": "A"}})
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 1,
            "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0}],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)
    contract = payload["training_launch_contract"]
    drain_buffer = contract["pretraining_drain_buffer"]

    assert contract["launch_allowed"] is False
    assert drain_buffer["status"] == "blocked_by_backpressure_gate"
    assert drain_buffer["writer"]["completed_lock_handoff_needed"] is True
    assert drain_buffer["recommended_command"][-3:] == ["--apply", "--handoff-only", "--json"]
    assert contract["recommended_prep_commands"][0][-3:] == ["--apply", "--handoff-only", "--json"]
    assert "backpressure_overload_severe" in contract["launch_blockers"]


def test_training_runtime_control_prefers_current_state_after_fast_handoff(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    writer_cycle = {
        "overall_status": "handoff_released",
        "writer_state_before": {
            "active": True,
            "active_source": "completed_lock_handoff_needed",
            "running": False,
            "status": "ok",
            "current_step": "complete",
            "writer_lock_held": True,
            "complete_lock_handoff_needed": True,
            "child_writer_active": False,
            "completed_shard_count": 6,
            "planned_shard_count": 6,
        },
        "writer_state_after_wait": {
            "active": False,
            "active_source": "idle",
            "running": False,
            "status": "ok",
            "current_step": "complete",
            "writer_lock_held": False,
            "complete_lock_handoff_needed": False,
            "child_writer_active": False,
            "completed_shard_count": 6,
            "planned_shard_count": 6,
        },
        "summary": {
            "completed_writer_lock_handoff_needed": True,
            "completed_writer_lock_handoff_released": True,
        },
    }
    backpressure_gate = {
        "severe": True,
        "pending_lines": 3096,
        "oldest_pending_age_seconds": 279.0,
        "pending_lines_threshold": 15000,
        "oldest_age_threshold_seconds": 240.0,
    }

    buffer = src._build_pretraining_drain_buffer(
        project_root=project_root,
        backpressure_gate=backpressure_gate,
        writer_cycle=writer_cycle,
        backlog_accelerator={"overall_status": "ready", "bulletproof_score": {"score": 95, "letter": "A"}},
    )

    assert buffer["status"] == "blocked_by_backpressure_gate"
    assert buffer["writer"]["completed_lock_handoff_needed"] is False
    assert buffer["recommended_command"][-1:] == ["--json"]
    assert "--handoff-only" not in buffer["recommended_command"]


def test_training_runtime_control_blocks_when_host_headroom_reserves_foreground_app(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "soft_guard", "decision": "cooldown_probe_only", "recommended_p_core_worker_cap": 4},
            "multitasking_headroom": {"active": True, "level": "interactive_developer", "open_apps": ["PyCharm"], "training_allowed_by_multitasking": False},
            "reopen_gate": {"safe_for_training": False, "training_blocked_by_multitasking": True},
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 1, "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0}]},
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is False
    assert contract["host_training_headroom_gate"]["status"] == "blocked"
    assert contract["host_training_headroom_gate"]["open_apps"] == ["PyCharm"]
    assert "host_training_headroom_not_clear" in contract["launch_blockers"]
    assert "host_multitasking_reserve_active" in contract["launch_blockers"]


def test_training_runtime_control_allows_two_bot_small_canary_when_governor_reopens_small_lane(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "needs_attention", "training_quality_score": 79.0, "top_priorities": ["lane_specific_training"]})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "soft_guard", "decision": "cooldown_probe_only", "recommended_p_core_worker_cap": 4},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": False,
                "small_canary_training_safe": True,
                "small_batch_training_safe": True,
                "training_batch_cap": 2,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_small_canary",
                    "reentry_gate": {"max_parallel_trainings": 2},
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 3,
            "seed_queue": [
                {"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0},
                {"bot_id": "brain_refinery_v50_investment_drawdown_risk", "current_runs": 16, "runs_remaining": 0, "priority": 39.0},
                {"bot_id": "brain_refinery_v59_risk_sentinel", "current_runs": 9, "runs_remaining": 3, "priority": 38.0},
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is True
    assert contract["recommended_batch_size"] == 2
    assert contract["host_training_headroom_gate"]["small_batch_training_safe"] is True
    assert contract["recommended_retrain_command"][-3:] == ["--retrain-profile", "coverage_small_canary", "--skip-master-update"]
    assert "brain_refinery_v10_seasonal,brain_refinery_v50_investment_drawdown_risk" in contract["recommended_retrain_command"]


def test_training_runtime_control_allows_one_bot_micro_canary_under_bounded_compression_relief(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 100.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "compression_relief", "decision": "cool_compression_before_widening", "recommended_p_core_worker_cap": 3},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": False,
                "small_canary_training_safe": True,
                "compression_relief_micro_canary_safe": True,
                "training_batch_cap": 1,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_micro_canary",
                    "reentry_gate": {"max_parallel_trainings": 1},
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 1,
            "seed_queue": [
                {"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0},
                {"bot_id": "brain_refinery_v50_investment_drawdown_risk", "current_runs": 16, "runs_remaining": 0, "priority": 39.0},
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is True
    assert contract["recommended_batch_size"] == 1
    assert contract["host_training_headroom_gate"]["memory_status"] == "compression_relief"
    assert contract["host_training_headroom_gate"]["batch_cap"] == 1
    assert "host_memory_relief_active" not in contract["launch_blockers"]
    assert contract["recommended_retrain_command"][-3:] == ["--retrain-profile", "coverage_micro_canary", "--skip-master-update"]
    assert "brain_refinery_v10_seasonal" in contract["recommended_retrain_command"]


def test_training_runtime_control_uses_repair_first_pool_for_quality_recovery_micro_canary(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 1200, "sequence_count": 160})
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "training_quality_score": 97.0,
            "top_priorities": ["ingestion_drain_time_guard", "promotion_coverage", "runtime_input_coverage"],
            "targeted_actions": {"targeted_retrain_bot_ids": []},
        },
    )
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}, "failure_details": []})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": False, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "compression_relief", "decision": "cool_compression_before_widening", "recommended_p_core_worker_cap": 3},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": False,
                "small_canary_training_safe": True,
                "compression_relief_micro_canary_safe": True,
                "training_batch_cap": 1,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_micro_canary",
                    "reentry_gate": {"max_parallel_trainings": 1},
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 2,
            "seed_queue": [
                {"bot_id": "brain_refinery_v10_seasonal", "current_runs": 0, "runs_remaining": 12, "priority": 43.0, "actions": ["targeted_retrain"]},
                {"bot_id": "brain_refinery_v13_choppy", "current_runs": 0, "runs_remaining": 12, "priority": 42.0, "actions": ["targeted_retrain"]},
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is True
    assert contract["training_quality_recovery_canary"] is True
    assert "training_quality_blocked" not in contract["launch_blockers"]
    assert contract["available_canary_pool_size"] == 0
    assert contract["available_repair_first_pool_size"] == 2
    assert contract["effective_launch_pool_size"] == 2
    assert contract["recommended_batch_size"] == 1
    assert contract["canary_batch"][0]["bot_id"] == "brain_refinery_v10_seasonal"
    assert contract["recommended_retrain_command"][-3:] == ["--retrain-profile", "coverage_micro_canary", "--skip-master-update"]


def test_training_runtime_control_uses_reentry_gate_when_outer_budget_is_stale(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 120, "sequence_count": 20})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 100.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "foreground_headroom", "decision": "preserve_user_app_headroom", "recommended_p_core_worker_cap": 5},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": False,
                "small_canary_training_safe": True,
                "small_batch_training_safe": True,
                "training_batch_cap": 2,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": False,
                    "profile": "coverage_micro_canary",
                    "watchdog_training_blocked": False,
                    "reentry_gate": {
                        "allowed": True,
                        "profile": "coverage_micro_canary",
                        "max_parallel_trainings": 1,
                        "blockers": [],
                    },
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 1, "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0}]},
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)
    contract = payload["training_launch_contract"]

    assert payload["host_training_headroom_gate"]["governor_reentry_gate_allowed"] is True
    assert payload["host_training_headroom_gate"]["governor_training_allowed"] is True
    assert contract["launch_allowed"] is True
    assert contract["recommended_batch_size"] == 1
    assert contract["recommended_retrain_command"][-3:] == ["--retrain-profile", "coverage_micro_canary", "--skip-master-update"]


def test_training_runtime_control_prefers_batch20_wave_lane_under_bounded_compression_relief(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 1200, "sequence_count": 160})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 100.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "compression_relief", "decision": "cool_compression_before_widening", "recommended_p_core_worker_cap": 3},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": False,
                "small_canary_training_safe": True,
                "compression_relief_micro_canary_safe": True,
                "batch20_training_safe": True,
                "batch20_execution_mode": "sequential_memory_guarded_waves",
                "batch20_wave_size": 3,
                "batch20_requires_between_target_memory_recheck": True,
                "training_batch_cap": 20,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_batch20_canary",
                    "reentry_gate": {"max_parallel_trainings": 20},
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 20,
            "seed_queue": [
                {"bot_id": f"brain_refinery_v{i}_compression_wave", "current_runs": 0, "runs_remaining": 12, "priority": 120.0 - i}
                for i in range(1, 25)
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root, limit=20)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is True
    assert contract["recommended_batch_size"] == 20
    assert contract["host_training_headroom_gate"]["memory_status"] == "compression_relief"
    assert contract["host_training_headroom_gate"]["selected_training_profile"] == "coverage_batch20_canary"
    assert contract["host_training_headroom_gate"]["batch20_wave_size"] == 3
    assert contract["recommended_retrain_command"][-3:] == ["--retrain-profile", "coverage_batch20_canary", "--skip-master-update"]


def test_training_runtime_control_allows_batch10_canary_when_host_and_governor_clear(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 500, "sequence_count": 80})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {"status": "clear", "decision": "safe_to_widen_after_soak", "recommended_p_core_worker_cap": 6},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": True,
                "batch10_training_safe": True,
                "batch20_training_safe": False,
                "training_batch_cap": 10,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_batch10_canary",
                    "reentry_gate": {"max_parallel_trainings": 10},
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 12,
            "seed_queue": [
                {"bot_id": f"brain_refinery_v{i}_batch_candidate", "current_runs": 0, "runs_remaining": 12, "priority": 100.0 - i}
                for i in range(1, 13)
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root, limit=20)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is True
    assert contract["recommended_batch_size"] == 10
    assert contract["host_training_headroom_gate"]["selected_training_profile"] == "coverage_batch10_canary"
    assert contract["recommended_retrain_command"][-3:] == ["--retrain-profile", "coverage_batch10_canary", "--skip-master-update"]


def test_training_runtime_control_allows_batch20_memory_guarded_canary(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 1200, "sequence_count": 160})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "heating_guard", "decision": "hold_while_memory_heats", "recommended_p_core_worker_cap": 4},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": False,
                "batch10_training_safe": True,
                "batch20_training_safe": True,
                "batch20_execution_mode": "sequential_memory_guarded_waves",
                "batch20_wave_size": 4,
                "batch20_requires_between_target_memory_recheck": True,
                "training_batch_cap": 20,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_batch20_canary",
                    "reentry_gate": {"max_parallel_trainings": 20},
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 20,
            "seed_queue": [
                {"bot_id": f"brain_refinery_v{i}_batch20_candidate", "current_runs": 0, "runs_remaining": 12, "priority": 120.0 - i}
                for i in range(1, 25)
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root, limit=20)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is True
    assert contract["recommended_batch_size"] == 20
    assert contract["host_training_headroom_gate"]["batch20_execution_mode"] == "sequential_memory_guarded_waves"
    assert contract["host_training_headroom_gate"]["selected_training_profile"] == "coverage_batch20_canary"
    assert contract["recommended_retrain_command"][-3:] == ["--retrain-profile", "coverage_batch20_canary", "--skip-master-update"]


def test_training_runtime_control_fills_batch20_from_bot_needs_topoff(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 1200, "sequence_count": 160})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {"status": "clear", "decision": "deep_green_wave_training", "recommended_p_core_worker_cap": 6},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": False,
                "batch10_training_safe": True,
                "batch20_training_safe": True,
                "batch20_execution_mode": "sequential_memory_guarded_waves",
                "batch20_wave_size": 4,
                "batch20_requires_between_target_memory_recheck": True,
                "training_batch_cap": 20,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_batch20_canary",
                    "reentry_gate": {"max_parallel_trainings": 20},
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 8,
            "seed_queue": [
                {"bot_id": f"brain_refinery_v{i}_coverage_seed", "current_runs": 8, "runs_remaining": 4, "priority": 110.0 - i}
                for i in range(1, 9)
            ],
        },
    )
    _write_json(
        health / "bot_needs_intelligence_latest.json",
        {
            "overall_status": "needs_action",
            "next_batches": {"training_topoff": [f"brain_refinery_v{i}_needs_topoff" for i in range(9, 29)]},
            "bot_needs": [
                {
                    "bot_id": f"brain_refinery_v{i}_needs_topoff",
                    "primary_need": "top_off_walk_forward_runs",
                    "priority": 82.0,
                    "evidence": {"walk_forward_runs": 9, "walk_forward_runs_remaining": 3},
                }
                for i in range(9, 29)
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root, limit=20)

    contract = payload["training_launch_contract"]
    assert payload["bot_needs"]["training_topoff_candidates"] == 20
    assert contract["launch_allowed"] is True
    assert contract["available_canary_pool_size"] >= 20
    assert contract["recommended_batch_size"] == 20
    assert any("brain_refinery_v20_needs_topoff" in part for part in contract["recommended_retrain_command"])


def test_training_runtime_control_allows_guarded_recovery_batch_when_quality_is_blocked(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 1200, "sequence_count": 160})
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "training_quality_score": 56.5,
            "top_priorities": ["runtime_input_coverage", "active_probation_isolation"],
            "targeted_actions": {"targeted_retrain_bot_ids": [f"brain_refinery_v{i}_recovery" for i in range(1, 21)]},
        },
    )
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {"status": "clear", "decision": "deep_green_wave_training", "recommended_p_core_worker_cap": 6},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": True,
                "batch10_training_safe": True,
                "batch20_training_safe": True,
                "batch20_execution_mode": "sequential_memory_guarded_waves",
                "batch20_wave_size": 4,
                "batch20_requires_between_target_memory_recheck": True,
                "training_batch_cap": 20,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_batch20_canary",
                    "reentry_gate": {"max_parallel_trainings": 20},
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 20,
            "seed_queue": [
                {"bot_id": f"brain_refinery_v{i}_recovery", "current_runs": 8, "runs_remaining": 4, "priority": 110.0 - i}
                for i in range(1, 21)
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root, limit=20)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is True
    assert contract["training_quality_recovery_canary"] is True
    assert "training_quality_blocked" not in contract["launch_blockers"]
    assert contract["recommended_batch_size"] == 20
    assert "skip-master-update" in " ".join(contract["recommended_retrain_command"])


def test_training_runtime_control_launches_weekend_soft_guard_recovery_batch(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 1200, "sequence_count": 160})
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "training_quality_score": 59.0,
            "top_priorities": ["runtime_input_coverage", "feature_store_lineage"],
            "targeted_actions": {"targeted_retrain_bot_ids": [f"brain_refinery_v{i}_weekend_recovery" for i in range(1, 21)]},
        },
    )
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green", "swap_used_gb": 2.58})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "soft_guard", "decision": "weekend_guarded_wave_training", "recommended_p_core_worker_cap": 4},
            "multitasking_headroom": {
                "active": True,
                "level": "media_playback",
                "open_apps": ["Music"],
                "training_allowed_by_multitasking": True,
                "training_max_parallel_trainings": 30,
                "weekend_media_training_window": True,
            },
            "reopen_gate": {
                "safe_for_training": False,
                "batch10_training_safe": True,
                "batch20_training_safe": True,
                "batch20_execution_mode": "sequential_memory_guarded_waves",
                "batch20_wave_size": 4,
                "batch20_requires_between_target_memory_recheck": True,
                "batch30_training_safe": True,
                "batch30_execution_mode": "sequential_memory_guarded_waves",
                "batch30_wave_size": 4,
                "batch30_requires_between_target_memory_recheck": True,
                "weekend_large_batch_window": True,
                "training_batch_cap": 30,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_batch30_canary",
                    "reentry_gate": {"max_parallel_trainings": 30},
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 20,
            "seed_queue": [
                {"bot_id": f"brain_refinery_v{i}_weekend_recovery", "current_runs": 8, "runs_remaining": 4, "priority": 110.0 - i}
                for i in range(1, 21)
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root, limit=5)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is True
    assert contract["training_quality_recovery_canary"] is True
    assert "training_quality_blocked" not in contract["launch_blockers"]
    assert contract["host_training_headroom_gate"]["selected_training_profile"] == "coverage_batch30_canary"
    assert contract["host_batch_cap"] == 30
    assert contract["recommended_batch_size"] == 5
    assert contract["recommended_retrain_command"][-3:] == ["--retrain-profile", "coverage_batch30_canary", "--skip-master-update"]


def test_training_runtime_control_treats_support_maintenance_freeze_as_training_advisory(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    bot_ids = [f"brain_refinery_v{i}_maintenance_advisory" for i in range(1, 11)]
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 1200, "sequence_count": 160})
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "ready",
            "training_quality_score": 94.0,
            "top_priorities": [],
            "targeted_actions": {"targeted_retrain_bot_ids": bot_ids},
        },
    )
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(
        health / "resource_guard_latest.json",
        {
            "resource_guard_ok": False,
            "resource_guard_reasons": [src.SUPPORT_MAINTENANCE_FREEZE_REASON],
            "memory_pressure_state": "green",
            "memory_pressure_kind": "normal",
            "swap_used_gb": 2.4,
        },
    )
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {"status": "clear", "decision": "deep_green_wave_training", "recommended_p_core_worker_cap": 6},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": True,
                "batch10_training_safe": True,
                "training_batch_cap": 10,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_batch10_canary",
                    "reentry_gate": {"max_parallel_trainings": 10},
                }
            }
        },
    )
    _write_json(walk / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0, "seed_queue": []})

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root, limit=10)

    gate = payload["resource_guard_training_gate"]
    contract = payload["training_launch_contract"]
    assert payload["overall_status"] == "ready"
    assert payload["resource_guard"]["ok"] is False
    assert payload["resource_guard"]["training_ok"] is True
    assert gate["raw_ok"] is False
    assert gate["training_ok"] is True
    assert gate["advisory_only"] is True
    assert "resource_guard_not_green" not in contract["launch_blockers"]
    assert contract["launch_allowed"] is True
    assert contract["recommended_batch_size"] == 10


def test_training_runtime_control_treats_active_creative_session_as_micro_canary_advisory(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    bot_id = "brain_refinery_v1_creative_session_canary"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 1200, "sequence_count": 160})
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "ready",
            "training_quality_score": 94.0,
            "top_priorities": [],
            "targeted_actions": {"targeted_retrain_bot_ids": [bot_id]},
        },
    )
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(
        health / "resource_guard_latest.json",
        {
            "resource_guard_ok": False,
            "resource_guard_reasons": [src.CREATIVE_SESSION_ACTIVE_REASON],
            "memory_pressure_state": "green",
            "memory_pressure_kind": "normal",
            "creative_session_level": "active",
        },
    )
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {"status": "foreground_headroom", "decision": "preserve_user_app_headroom", "recommended_p_core_worker_cap": 3},
            "multitasking_headroom": {
                "active": True,
                "level": "interactive_developer",
                "open_apps": ["PyCharm", "Google Chrome"],
                "training_allowed_by_multitasking": True,
            },
            "reopen_gate": {
                "safe_for_training": False,
                "small_canary_training_safe": True,
                "training_batch_cap": 1,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_micro_canary",
                    "reentry_gate": {"max_parallel_trainings": 1},
                }
            }
        },
    )
    _write_json(walk / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0, "seed_queue": []})

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root, limit=10)

    gate = payload["resource_guard_training_gate"]
    contract = payload["training_launch_contract"]
    assert payload["overall_status"] == "ready"
    assert payload["resource_guard"]["ok"] is False
    assert payload["resource_guard"]["training_ok"] is True
    assert gate["raw_ok"] is False
    assert gate["training_ok"] is True
    assert gate["advisory_only"] is True
    assert "resource_guard_not_green" not in contract["launch_blockers"]
    assert contract["launch_allowed"] is True
    assert contract["host_training_headroom_gate"]["selected_training_profile"] == "coverage_micro_canary"
    assert contract["recommended_batch_size"] == 1


def test_training_runtime_control_blocks_batch20_when_storage_quota_hard_breaches(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 1200, "sequence_count": 160})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False}})
    _write_json(
        health / "storage_quota_guard_latest.json",
        {
            "overall_status": "blocked",
            "quota_summary": {
                "hard_breaches": 1,
                "soft_breaches": 0,
                "blocked_families": ["decisions"],
                "worst_over_hard_gb": 9.431,
                "worst_hard_ratio": 1.262,
            },
            "recommended_actions": ["keep expansion and heavy training gated until blocked storage quota lanes fall below hard quota"],
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {"status": "clear", "decision": "deep_green_wave_training", "recommended_p_core_worker_cap": 6},
            "multitasking_headroom": {"active": False, "level": "background_available", "open_apps": [], "training_allowed_by_multitasking": True},
            "reopen_gate": {
                "safe_for_training": True,
                "batch10_training_safe": True,
                "batch20_training_safe": True,
                "batch20_execution_mode": "sequential_memory_guarded_waves",
                "batch20_wave_size": 4,
                "batch20_requires_between_target_memory_recheck": True,
                "training_batch_cap": 20,
            },
        },
    )
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "budgets": {
                "training": {
                    "allowed": True,
                    "profile": "coverage_batch20_canary",
                    "reentry_gate": {"max_parallel_trainings": 20},
                }
            }
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {
            "coverage_shortfall_bots": 20,
            "seed_queue": [
                {"bot_id": f"brain_refinery_v{i}_quota_blocked", "current_runs": 8, "runs_remaining": 4, "priority": 110.0 - i}
                for i in range(1, 21)
            ],
        },
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root, limit=20)

    contract = payload["training_launch_contract"]
    assert contract["launch_allowed"] is False
    assert contract["recommended_batch_size"] == 0
    assert "storage_quota_hard_breach" in contract["launch_blockers"]
    assert contract["storage_quota_training_gate"]["blocked_families"] == ["decisions"]
    assert any(command[-2:] == ["storage-quota-guard", "--json"] for command in contract["recommended_prep_commands"])


def test_training_runtime_control_uses_storage_backpressure_over_stale_health_gate(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(health / "health_gates_latest.json", {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False, "backpressure_pending_lines": 0}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 12.5,
            "backpressure": {
                "total_pending_lines": 87605,
                "oldest_pending_age_seconds": 9962.0,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(health / "backpressure_super_drainer_latest.json", {"summary": {"final_pending_lines": 87605}})
    _write_json(
        walk / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 1, "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0}]},
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["backpressure_training_gate"]["pending_lines"] == 87605
    assert payload["backpressure_training_gate"]["cooling_down"] is False
    assert "ingestion_storage_control" in payload["backpressure_training_gate"]["sources"]
    assert payload["training_launch_contract"]["launch_allowed"] is False
    assert "backpressure_overload_severe" in payload["training_launch_contract"]["launch_blockers"]


def test_training_runtime_control_trusts_downward_reconciled_storage_overlay(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(
        health / "health_gates_latest.json",
        {"recommended_operating_mode": "normal", "inputs": {"backpressure_overload_severe": False, "backpressure_pending_lines": 251486}},
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"pending_lines": 3863, "pending_lines_total": 251486, "oldest_pending_age_seconds": 0.013},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.306,
            "backpressure": {
                "overlay_adjusted": True,
                "total_pending_lines": 1290,
                "oldest_pending_age_seconds": 73.346,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
            },
            "sql_ingestion_pending_overlay": {
                "used_for_pressure": True,
                "reconciled_downward_for_pressure": True,
            },
        },
    )
    _write_json(health / "backpressure_super_drainer_latest.json", {"summary": {"final_pending_lines": 251486}})
    _write_json(
        walk / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 0, "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0}]},
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    assert payload["backpressure_training_gate"]["pending_lines"] == 1290
    assert payload["backpressure_training_gate"]["severe"] is False
    assert "sql_overlay_reconciled_downward" in payload["backpressure_training_gate"]["sources"]
    assert "backpressure_overload_severe" not in payload["training_launch_contract"]["launch_blockers"]


def test_training_runtime_control_clears_stale_severe_flags_when_storage_is_numeric_green(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(
        health / "health_gates_latest.json",
        {
            "recommended_operating_mode": "normal",
            "inputs": {
                "backpressure_overload_severe": True,
                "backpressure_pending_lines": 251486,
                "sql_progress_status": "running",
            },
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"overload": True, "pending_lines_total": 251486, "oldest_pending_age_seconds": 409.0},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "needs_work",
            "severity": "high",
            "pressure_index": 0.075,
            "backpressure": {
                "overlay_adjusted": True,
                "total_pending_lines": 47,
                "oldest_pending_age_seconds": 17.931,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
            },
            "sql_ingestion_pending_overlay": {
                "used_for_pressure": True,
                "reconciled_downward_for_pressure": True,
            },
        },
    )
    _write_json(health / "backpressure_super_drainer_latest.json", {"summary": {"final_pending_lines": 47}})
    _write_json(
        walk / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 0, "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0}]},
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    assert payload["backpressure_training_gate"]["pending_lines"] == 47
    assert payload["backpressure_training_gate"]["severe"] is False
    assert payload["backpressure_training_gate"]["storage_numeric_clear"] is True
    assert payload["backpressure_training_gate"]["stale_health_severe_ignored"] is True
    assert payload["backpressure_training_gate"]["stale_ingestion_overload_ignored"] is True
    assert "backpressure_overload_severe" not in payload["training_launch_contract"]["launch_blockers"]


def test_training_runtime_control_trusts_health_gate_storage_override_over_stale_raw_storage(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(
        health / "health_gates_latest.json",
        {
            "recommended_operating_mode": "normal",
            "inputs": {
                "backpressure_overload_severe": False,
                "backpressure_pending_lines": 0,
                "backpressure_oldest_pending_age_seconds": 0.0,
                "sql_progress_status": "complete",
                "backpressure_storage_control_override": {
                    "active": True,
                    "source": "fresh_empty_sql_ingestion_overlay",
                    "age_seconds": 72.0,
                    "pending_lines": 0,
                    "pending_lines_total": 0,
                    "oldest_pending_age_seconds": 0.0,
                    "overload": False,
                    "overlay_clear": True,
                    "queue_clear": True,
                },
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
            {
                "overall_status": "blocked",
                "severity": "blocked",
                "pressure_index": 259.871,
            "backpressure": {
                "total_pending_lines": 12585,
                "oldest_pending_age_seconds": 62368.944,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(
        walk / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 0, "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0}]},
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    assert payload["backpressure_training_gate"]["pending_lines"] == 0
    assert payload["backpressure_training_gate"]["oldest_pending_age_seconds"] == 0.0
    assert payload["backpressure_training_gate"]["pressure_index"] == 0.0
    assert payload["backpressure_training_gate"]["raw_pressure_index"] > 1.0
    assert payload["backpressure_training_gate"]["storage_control_override_clear"] is True
    assert payload["backpressure_training_gate"]["severe"] is False
    assert "health_gate_storage_control_override" in payload["backpressure_training_gate"]["sources"]
    assert "backpressure_overload_severe" not in payload["training_launch_contract"]["launch_blockers"]


def test_training_runtime_control_trusts_stable_needs_work_storage_over_stale_super_drainer(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 100.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(
        health / "health_gates_latest.json",
        {
            "recommended_operating_mode": "normal",
            "inputs": {
                "backpressure_overload_severe": False,
                "backpressure_pending_lines": 456,
                "backpressure_oldest_pending_age_seconds": 0.96,
            },
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"pending_lines_total": 251486, "oldest_pending_age_seconds": 409.0},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "needs_work",
            "severity": "stable",
            "pressure_index": 0.03,
            "backpressure": {
                "total_pending_lines": 784,
                "oldest_pending_age_seconds": 0.96,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
            },
            "writer_shedding": {
                "hard_breaches": [],
                "elevated_breaches": ["core"],
                "target_breaches": ["core"],
            },
        },
    )
    _write_json(health / "backpressure_super_drainer_latest.json", {"summary": {"final_pending_lines": 275203}})
    _write_json(
        walk / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 0, "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0}]},
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    assert payload["backpressure_training_gate"]["pending_lines"] == 784
    assert payload["backpressure_training_gate"]["severe"] is False
    assert payload["backpressure_training_gate"]["storage_status_authoritative"] is True
    assert payload["backpressure_training_gate"]["storage_live_authoritative"] is True
    assert "backpressure_overload_severe" not in payload["training_launch_contract"]["launch_blockers"]


def test_training_runtime_control_uses_ready_storage_truth_over_stale_old_age(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": _fresh_ts(), "row_count": 100, "sequence_count": 12})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 91.0, "top_priorities": []})
    _write_json(health / "retrain_scorecard_latest.json", {"retry_pack": {"command": []}})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": True, "failure_details": []})
    _write_json(health / "resource_guard_latest.json", {"resource_guard_ok": True, "memory_pressure_state": "green"})
    _write_json(
        health / "health_gates_latest.json",
        {
            "recommended_operating_mode": "normal",
            "inputs": {
                "backpressure_overload_severe": True,
                "backpressure_pending_lines": 12006,
                "backpressure_oldest_pending_age_seconds": 4928.503,
                "sql_progress_status": "running",
            },
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"overload": True, "pending_lines_total": 12006, "oldest_pending_age_seconds": 4928.503},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.04,
            "backpressure": {
                "total_pending_lines": 12006,
                "oldest_pending_age_seconds": 0.0,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
            },
        },
    )
    _write_json(health / "backpressure_super_drainer_latest.json", {"summary": {"final_pending_lines": 12006}})
    _write_json(
        walk / "coverage_seed_latest.json",
        {"coverage_shortfall_bots": 0, "seed_queue": [{"bot_id": "brain_refinery_v10_seasonal", "current_runs": 14, "runs_remaining": 0, "priority": 40.0}]},
    )

    runtime_python = project_root / ".venv312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(src, "resolve_runtime_python", lambda _root: runtime_python)

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps(
                {
                    "python": "3.12.12",
                    "platform": "macOS",
                    "modules": {"mlx": True, "torch": True, "onnxruntime": False, "tensorflow": False, "jax": False},
                }
            )
            self.stderr = ""

    monkeypatch.setattr(src.subprocess, "run", lambda *args, **kwargs: _Proc())

    payload = src.build_payload(project_root)

    assert payload["backpressure_training_gate"]["pending_lines"] == 12006
    assert payload["backpressure_training_gate"]["oldest_pending_age_seconds"] == 0.0
    assert payload["backpressure_training_gate"]["storage_live_authoritative"] is True
    assert payload["backpressure_training_gate"]["severe"] is False
    assert "backpressure_overload_severe" not in payload["training_launch_contract"]["launch_blockers"]
