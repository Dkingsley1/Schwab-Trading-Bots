from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import infrastructure_autofix_bot as infra_src
from scripts.ops import storage_reconnect_infrabot as bot_src
from scripts.ops import storage_reconnect_regression_guard as guard_src


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_storage_reconnect_regression_guard_contract_is_ready() -> None:
    payload = guard_src.build_payload(PROJECT_ROOT, check_launchd=False, check_swift_parse=False)

    assert payload["contract_ok"] is True
    assert payload["missing_contracts"] == []
    assert payload["regression_guard_contract"]["requires_split_brain_reconcile"] is True
    assert payload["regression_guard_contract"]["requires_storage_pressure_clearance"] is True
    assert payload["regression_guard_contract"]["requires_global_halt_auto_clear"] is True


def test_storage_reconnect_infrabot_plans_safe_repairs(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    (project_root / "scripts" / "ops").mkdir(parents=True)
    (project_root / "scripts" / "install_storage_eject_guard_launchd.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    (project_root / "scripts" / "ops" / "opsctl.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    _write_json(
        health / "storage_reconnect_regression_guard_latest.json",
        '{"overall_status":"degraded","contract_ok":true,"automation":{"launchd":{"running":false,"plist_exists":false}},"live_recovery":{"split_brain_unresolved_conflicts":1,"total_pending_lines":50000}}\n',
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        '{"overall_status":"blocked","backpressure":{"total_pending_lines":50000}}\n',
    )
    _write_json(
        health / "global_risk_killswitch_latest.json",
        '{"clear_blockers":["write_path_recovery_pending"]}\n',
    )
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        '{"overall_status":"degraded","queue_depth":50000}\n',
    )
    monkeypatch.setattr(bot_src, "_guard_payload", lambda project_root, timeout_sec: guard_src.load_json(health / "storage_reconnect_regression_guard_latest.json"))

    payload = bot_src.build_payload(project_root, apply=False)

    names = [row["name"] for row in payload["repair_plan"]]
    assert "install_storage_eject_guard_launchd" in names
    assert "split_brain_reconcile" in names
    assert "storage_pressure_clearance" in names
    assert "global_halt_safe_refresh" in names
    assert "global_halt_safe_auto_clear" in names
    assert payload["metrics"]["repair_plan_count"] == len(names)


def test_infrastructure_autofix_assigns_storage_reconnect_infrabot(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "storage_reconnect_infrabot_latest.json",
        '{"overall_status":"degraded","metrics":{"repair_plan_count":2}}\n',
    )
    _write_json(
        health / "storage_reconnect_regression_guard_latest.json",
        '{"overall_status":"ready","metrics":{"missing_contract_count":0}}\n',
    )
    monkeypatch.setattr(
        infra_src,
        "_run_json",
        lambda cmd, *, cwd, timeout_sec: {
            "cmd": list(cmd),
            "rc": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
            "payload": {"overall_status": "ready", "ok": True, "metrics": {}},
        },
    )

    payload = infra_src.build_payload(project_root, apply=False)

    names = [row["name"] for row in payload["repair_plan"]]
    assert "storage_reconnect_infrabot" in names
    assert "storage_reconnect_infrabot" in payload["infra_bots"]
    assert payload["metrics"]["storage_reconnect_repair_plan_count"] == 2
