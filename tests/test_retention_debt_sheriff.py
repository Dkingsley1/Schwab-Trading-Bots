import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import retention_debt_sheriff as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_retention_debt_sheriff_filters_to_priority_hot_shards_and_delegates_when_writer_active(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        src.maintenance_src,
        "_priority_retention_focus",
        lambda *args, **kwargs: {
            "enabled": True,
            "priority_rows": [
                {
                    "shard": "shadow_attribution",
                    "retention_debt_gb": 8.0,
                    "latency_limit_multiplier": 1.05,
                    "storage_breached": True,
                    "latency_breached": False,
                    "recommended_action": "force_retention_and_throttle",
                },
                {
                    "shard": "explanations",
                    "retention_debt_gb": 51.055,
                    "latency_limit_multiplier": 1.727,
                    "storage_breached": True,
                    "latency_breached": True,
                    "recommended_action": "force_retention_and_throttle",
                },
                {
                    "shard": "crypto_explanations",
                    "retention_debt_gb": 12.71,
                    "latency_limit_multiplier": 1.17,
                    "storage_breached": True,
                    "latency_breached": True,
                    "recommended_action": "force_retention_and_throttle",
                },
            ],
        },
    )
    monkeypatch.setattr(src.coordinator_src, "writer_state_snapshot", lambda *args, **kwargs: {"active": True, "current_step": "merge_primary"})

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "writer_cycle_coordinator.py" in joined:
            payload = {"overall_status": "applied", "summary": {"maintenance_applied": True}}
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "blocked"}
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "degraded"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "degraded"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 9.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "applied"
    assert payload["focus"]["focus_shards"] == ["explanations", "crypto_explanations", "shadow_attribution"]
    assert payload["summary"]["targeted_retention_debt_gb"] == 71.765
    assert payload["steps"]["writer_cycle_coordinator"]["status"] == "ok"
    assert payload["refresh_steps"]["operator_cockpit"]["status"] == "ok"
