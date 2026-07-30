import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import infrastructure_autofix_bot as src


READY_STAMP = "2099-04-23T20:00:00+00:00"
STALE_STAMP = "2020-04-23T20:00:00+00:00"


def _ready_payload(raw_path: str | Path) -> dict[str, Any]:
    path = str(raw_path)
    payload: dict[str, Any] = {"timestamp_utc": READY_STAMP, "overall_status": "ready", "ok": True}
    if path.endswith("remote_alert_control_latest.json"):
        payload["channels"] = {"any_configured": True}
        payload["critical_backlog"] = {"unsent_count": 0}
    if path.endswith("runtime_snapshot_cache_control_latest.json"):
        payload["cache_health"] = {"snapshot_ready": True}
    if path.endswith("mlx_intelligence_router_latest.json"):
        payload["library_coverage"] = {"coverage_ratio": 1.0}
    if path.endswith("library_upgrade_route_control_latest.json"):
        payload["upgrade_plan"] = {"hard_blocker_count": 0}
    if path.endswith("stateful_storage_regression_guard_latest.json"):
        payload["metrics"] = {"local_stateful_gb": 0.0}
    if path.endswith("ingestion_storage_control_latest.json"):
        payload["backpressure"] = {"total_pending_lines": 0, "estimated_total_drain_minutes": 0.0}
        payload["storage"] = {"retention_debt_gb": 0.0}
    if path.endswith("auth_lease_manager_latest.json"):
        payload["lease_budget"] = {"expires_in_seconds": 3600}
    if path.endswith("artifact_freshness_slo_latest.json"):
        payload["sla_summary"] = {"stale_required": 0, "stale_optional": 0}
    if path.endswith("process_watchdog_latest.json"):
        payload["watchdog_intelligence"] = {"active_issue_count": 0}
    if path.endswith("stale_surface_autohealer_latest.json"):
        payload["metrics"] = {"planned_repair_count": 0}
    return payload


def _install_isolated_loaders(
    monkeypatch,
    payloads: dict[str, dict[str, Any]],
) -> None:
    def fake_load(project_root: Path, raw_path: str | Path) -> tuple[dict[str, Any], Path | None]:
        path = str(raw_path)
        return dict(payloads.get(path, _ready_payload(path))), project_root / path

    def fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "overall_status": "ready",
            "ok": True,
            "commands_changed": False,
            "runbook_changed": False,
            "metrics": {"blocked_entry_count": 0},
        }
        return {"cmd": list(cmd), "rc": 0, "timed_out": False, "timeout_sec": timeout_sec, "payload": payload}

    monkeypatch.setattr(src, "_load_freshest_json_with_path", fake_load)
    monkeypatch.setattr(src, "_run_json", fake_run_json)


def test_infrastructure_autofix_refreshes_health_gates_before_rechecking_halt_control(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    payloads = {
        "governance/health/health_gates_latest.json": {
            "timestamp_utc": STALE_STAMP,
            "overall_status": "ready",
            "ok": True,
            "hard_gate_triggered": False,
        },
        "governance/health/halt_trigger_control_plane_latest.json": {
            "timestamp_utc": READY_STAMP,
            "overall_status": "blocked",
            "effective_state": "safety_artifact_uncertain",
            "artifacts": {"health_gates": {"state": "stale"}},
            "blockers": {
                "halt_clear": ["critical_artifact_stale:health_gates"],
                "live_execution": ["paper_trade_lock_active", "critical_artifact_stale:health_gates"],
            },
        },
        "governance/health/coordination_state_latest.json": {
            "timestamp_utc": READY_STAMP,
            "overall_status": "guarded",
            "artifact_issues": [],
        },
    }
    _install_isolated_loaders(monkeypatch, payloads)

    payload = src.build_payload(project_root, apply=False, timeout_sec=120)
    names = [row["name"] for row in payload["repair_plan"]]

    assert names[:2] == ["health_gates_refresh", "halt_trigger_control_plane"]
    assert payload["metrics"]["health_gates_stale"] is True
    assert payload["overall_status"] == "degraded"


def test_infrastructure_autofix_treats_paper_only_live_order_lock_as_managed_when_health_gates_are_fresh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    payloads = {
        "governance/health/health_gates_latest.json": {
            "timestamp_utc": READY_STAMP,
            "overall_status": "ready",
            "ok": True,
            "hard_gate_triggered": False,
        },
        "governance/health/halt_trigger_control_plane_latest.json": {
            "timestamp_utc": READY_STAMP,
            "overall_status": "blocked",
            "effective_state": "live_read_only",
            "artifacts": {"health_gates": {"state": "fresh"}},
            "blockers": {
                "halt_clear": [],
                "live_execution": ["paper_trade_lock_active", "runtime_release_live_read_only"],
            },
            "execution_policy": {
                "effective_live_order_execution_allowed": False,
                "paper_trade_lock_active": True,
            },
        },
        "governance/health/coordination_state_latest.json": {
            "timestamp_utc": READY_STAMP,
            "overall_status": "guarded",
            "artifact_issues": [],
        },
    }
    _install_isolated_loaders(monkeypatch, payloads)

    payload = src.build_payload(project_root, apply=False, timeout_sec=120)

    assert payload["repair_plan"] == []
    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["health_gates_stale"] is False
