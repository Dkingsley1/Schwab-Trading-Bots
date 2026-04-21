import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import training_runtime_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_training_runtime_control_surfaces_runtime_backend_parity(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": "2026-04-21T15:00:00+00:00", "row_count": 10, "sequence_count": 2})
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


def test_training_runtime_control_blocks_on_missing_mlx_failure(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": "2026-04-21T15:00:00+00:00", "row_count": 10, "sequence_count": 2})
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
    _write_json(health / "runtime_training_snapshot_latest.json", {"timestamp_utc": "2026-04-21T15:00:00+00:00", "row_count": 10, "sequence_count": 2})
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
