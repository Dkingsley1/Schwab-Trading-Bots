import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/runtime_gate_dashboard.py")
spec = importlib.util.spec_from_file_location("runtime_gate_dashboard_contract", MODULE_PATH)
runtime_gate_dashboard = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(runtime_gate_dashboard)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_runtime_gate_dashboard_marks_missing_sections_with_explicit_contract_state(tmp_path):
    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["overall_status"] == payload["overall"]["status"]
    assert payload["ok"] == payload["overall"]["ok"]
    assert payload["runtime"]["artifact_status"] == "missing"
    assert payload["runtime"]["artifact_reason"] == "artifact_missing"
    assert payload["runtime"]["mode"] == "unknown"
    assert payload["apple_silicon"]["artifact_status"] == "missing"
    assert payload["memory"]["artifact_status"] == "missing"
    assert payload["training"]["artifact_status"] == "missing"
    assert payload["platform"]["artifact_status"] == "missing"


def test_runtime_gate_dashboard_manages_paper_soak_auth_warning_attention(tmp_path):
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {"overall_status": "ready", "overall_grade": "A+", "safe_to_leave_unattended": True},
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {"ok": True, "overall_status": "ready", "paper_armed": True, "paper_stage": "armed", "failed_guard_count": 0, "failed_guards": []},
    )
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only"},
            },
        },
    )
    auth_path = health / "auth_lease_manager_latest.json"
    supervisor_path = health / "schwab_auth_supervisor_latest.json"
    _write_json(
        auth_path,
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {"expires_in_seconds": 1120, "critical_lease_seconds": 600, "token_lease_grace": True},
            "broker_state": {"broker_ready": True, "broker_operable": True, "network_ok": True, "auth_ok": False, "auth_probe_ok": False},
        },
    )
    _write_json(supervisor_path, {"overall_status": "ready", "ok": True, "paper_soak_auth_operable": True})

    artifacts = {
        "auth_lease_manager": {"path": str(auth_path), "summary": {"overall_status": "degraded"}},
        "schwab_auth_supervisor": {"path": str(supervisor_path), "summary": {"overall_status": "ready"}},
    }
    context = runtime_gate_dashboard._dashboard_soak_context(tmp_path)
    reason = runtime_gate_dashboard._attention_managed_by_green_soak(
        "auth_lease_manager_needs_work",
        artifacts,
        context,
    )

    assert context["enabled"] is True
    assert context["guarded_health_ready"] is True
    assert reason == "schwab_auth_warning_managed_while_token_above_paper_readiness_floor"


def test_dashboard_soak_context_accepts_guarded_ready_health_fast(tmp_path):
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {"overall_status": "ready", "overall_grade": "A+", "safe_to_leave_unattended": True},
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "paper_armed": True,
            "paper_blocked": False,
            "failed_guard_count": 0,
            "failed_guards": [],
        },
    )
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "guarded_ready",
            "operational_readiness": {"guarded_paper": {"ok": True, "status": "ready"}},
        },
    )

    context = runtime_gate_dashboard._dashboard_soak_context(tmp_path)

    assert context["enabled"] is True
    assert context["guarded_health_ready"] is True


def test_runtime_gate_dashboard_manages_bounded_transient_backlog_attention(tmp_path):
    health = tmp_path / "governance" / "health"
    storage_path = health / "ingestion_storage_control_latest.json"
    storage_payload = {
        "overall_status": "ready",
        "severity": "elevated",
        "pressure_index": 0.926,
        "continuous_run_soak_contract": {
            "status": "blocked",
            "ready": False,
            "soak_ready": False,
            "blockers": ["steady_state_targets_not_clear"],
        },
        "bounded_recovery_contract": {
            "route_verified": True,
            "active_drain_progress": True,
            "drain_delta_signal_observed": True,
            "hard_gate_active": False,
            "effective_hard_gate_active": False,
        },
        "data_integrity": {
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_oversize_payloads": 0,
            "sql_overlay_ops_write_failures": 0,
        },
        "writer_shedding": {"hard_breaches": [], "elevated_breaches": []},
        "backpressure": {
            "raw_live": {
                "core_pending_lines": 3902,
                "total_pending_lines": 4916,
                "oldest_pending_age_seconds": 222.349,
            }
        },
    }
    _write_json(storage_path, storage_payload)
    artifacts = {
        "ingestion_storage_control": {
            "path": str(storage_path),
            "summary": {
                "overall_status": "ready",
                "severity": "elevated",
                "pressure_index": 0.926,
            },
        },
        "external_backlog_drain": {
            "summary": {
                "overall_status": "drain_active",
                "recommended_now": True,
                "aged_candidate_files": 0,
                "writer_busy": False,
            }
        },
    }

    assert runtime_gate_dashboard._ingestion_soak_ready_for_dashboard(artifacts) is True
    reason = runtime_gate_dashboard._attention_managed_by_green_soak(
        "external_backlog_drain_recommended",
        artifacts,
        {"enabled": True},
    )
    assert reason == "external_backlog_handoff_managed_while_ingestion_soak_is_green"
