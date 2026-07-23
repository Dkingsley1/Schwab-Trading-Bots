from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import live_canary_readiness_contract as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_ready_artifacts(project_root: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    health = project_root / "governance" / "health"
    _write_json(health / "paper_profitability_control_latest.json", {"timestamp_utc": now, "overall_status": "ready", "raw_profitability_grade": "A"})
    _write_json(health / "paper_runtime_profitability_controls_latest.json", {"timestamp_utc": now, "overall_status": "ready", "raw_profitability_grade": "A"})
    _write_json(health / "paper_execution_truth_layer_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True, "failed_checks": []})
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True, "failed_checks": []})
    _write_json(health / "paper_400_ramp_latest.json", {"timestamp_utc": now, "overall_status": "ready", "stage": "armed", "blockers": []})
    _write_json(health / "broker_readiness_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ready_for_open": True, "auth_ok": True, "network_ok": True, "token_expires_in_seconds": 2400})
    _write_json(health / "schwab_auth_supervisor_latest.json", {"timestamp_utc": now, "overall_status": "ready", "token": {"expires_in_seconds": 2400}})
    _write_json(health / "auth_lease_manager_latest.json", {"timestamp_utc": now, "overall_status": "ready", "lease_state": "healthy", "expires_in_seconds": 2400})
    _write_json(health / "ingestion_storage_control_latest.json", {"timestamp_utc": now, "overall_status": "ready", "pressure_index": 0.01, "backpressure": {"total_pending_lines": 0}})
    _write_json(health / "health_gates_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(health / "promotion_quality_gate_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(health / "promotion_readiness_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(health / "paper_performance_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})


def test_live_canary_readiness_contract_blocks_raw_d_grade(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _seed_ready_artifacts(project_root)
    _write_json(project_root / "governance" / "health" / "paper_profitability_control_latest.json", {"overall_status": "ready", "raw_profitability_grade": "D"})
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})

    payload = src.build_payload(project_root, out_path=project_root / "governance" / "health" / "live_canary_readiness_contract_latest.json")

    raw_gate = next(gate for gate in payload["gates"] if gate["gate_id"] == "raw_profitability_posture")
    assert payload["overall_status"] == "blocked"
    assert payload["live_money_canary_blocked"] is True
    assert raw_gate["ready"] is False
    assert "raw_profitability_hard_block_below_C" in raw_gate["blockers"]
    assert "no raw D-grade posture" in payload["infrastructure_message"]


def test_live_canary_readiness_contract_can_clear_after_sustained_window(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    out_path = project_root / "governance" / "health" / "live_canary_readiness_contract_latest.json"
    _seed_ready_artifacts(project_root)
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})
    _write_json(
        out_path,
        {
            "overall_status": "blocked",
            "continuous_all_gates_ready_since_utc": (datetime.now(timezone.utc) - timedelta(hours=170)).isoformat(),
        },
    )

    payload = src.build_payload(project_root, out_path=out_path)

    assert payload["overall_status"] == "ready"
    assert payload["live_canary_money_ready"] is True
    assert payload["sustained_window"]["sustained_window_met"] is True
    assert payload["ready_gate_count"] == payload["gate_count"]
