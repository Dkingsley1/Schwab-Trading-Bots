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
    assert payload["regression_guard_contract"]["requires_auto_failback_opt_in"] is True
    assert payload["regression_guard_contract"]["requires_local_override_mount_suppression"] is True
    assert payload["regression_guard_contract"]["requires_transactional_sqlite_local_failover"] is True
    assert payload["regression_guard_contract"]["requires_swift_semantic_typecheck"] is True
    assert payload["regression_guard_contract"]["requires_atomic_transition_state"] is True
    assert payload["regression_guard_contract"]["requires_standby_disconnect_restart_suppression"] is True
    assert payload["regression_guard_contract"]["requires_external_write_certification"] is True
    assert payload["regression_guard_contract"]["requires_compiled_runtime_binary"] is True


def test_storage_reconnect_guard_flags_external_sqlite_dependency_in_local_mode(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    scripts_ops = project_root / "scripts" / "ops"
    scripts_ops.mkdir(parents=True)
    guard_contract = "\n".join(guard_src.REQUIRED_GUARD_SNIPPETS.values())
    opsctl_contract = "\n".join(guard_src.REQUIRED_OPSCTL_SNIPPETS.values())
    (scripts_ops / "storage_eject_guard.swift").write_text(guard_contract, encoding="utf-8")
    (scripts_ops / "opsctl.sh").write_text(opsctl_contract, encoding="utf-8")
    (scripts_ops / "storage_sqlite_local_failover.py").write_text("# contract\n", encoding="utf-8")
    (scripts_ops / "run_storage_eject_guard_launchd.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    (project_root / "scripts" / "install_storage_eject_guard_launchd.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    health = project_root / "governance" / "health"
    _write_json(
        health / "storage_failback_sync_latest.json",
        '{"mode":"local_fallback","certified_mode":"local_fallback","sqlite_skip_report":{"entries":[{"relative_path":"data/jsonl_link.sqlite3","classification":"active_external_route"}]}}\n',
    )
    _write_json(health / "storage_mount_guard_latest.json", '{"external_available":true}\n')
    _write_json(health / "ingestion_storage_control_latest.json", '{"overall_status":"ready"}\n')

    payload = guard_src.build_payload(project_root, check_launchd=False, check_swift_parse=False)

    assert payload["contract_ok"] is True
    assert "local_mode_external_sqlite_route" in payload["live_recovery"]["blockers"]
    assert payload["live_recovery"]["external_sqlite_routes"] == ["data/jsonl_link.sqlite3"]
    assert "storage-sqlite-local-failover --apply" in " ".join(payload["recommended_actions"])


def test_storage_reconnect_guard_allows_intentional_local_hot_storage(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    scripts_ops = project_root / "scripts" / "ops"
    scripts_ops.mkdir(parents=True)
    (scripts_ops / "storage_eject_guard.swift").write_text(
        "\n".join(guard_src.REQUIRED_GUARD_SNIPPETS.values()), encoding="utf-8"
    )
    (scripts_ops / "opsctl.sh").write_text(
        "\n".join(guard_src.REQUIRED_OPSCTL_SNIPPETS.values()), encoding="utf-8"
    )
    (scripts_ops / "storage_sqlite_local_failover.py").write_text("# contract\n", encoding="utf-8")
    (scripts_ops / "run_storage_eject_guard_launchd.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    (project_root / "scripts" / "install_storage_eject_guard_launchd.sh").write_text(
        "#!/bin/zsh\n", encoding="utf-8"
    )
    health = project_root / "governance" / "health"
    _write_json(health / "storage_failback_sync_latest.json", '{"certified_mode":"local_fallback"}\n')
    _write_json(
        health / "storage_mount_guard_latest.json",
        '{"external_available":false,"external_required_for_hot_path":false,"probe_skipped_external_io":true}\n',
    )
    _write_json(health / "ingestion_storage_control_latest.json", '{"overall_status":"ready"}\n')

    payload = guard_src.build_payload(project_root, check_launchd=False, check_swift_parse=False)

    assert payload["overall_status"] == "ready"
    assert "external_mount_unavailable" not in payload["live_recovery"]["blockers"]
    assert payload["live_recovery"]["external_required_for_hot_path"] is False
    assert payload["live_recovery"]["external_probe_skipped"] is True


def test_storage_reconnect_guard_uses_ready_transition_for_external_availability(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    scripts_ops = project_root / "scripts" / "ops"
    scripts_ops.mkdir(parents=True)
    (scripts_ops / "storage_eject_guard.swift").write_text(
        "\n".join(guard_src.REQUIRED_GUARD_SNIPPETS.values()), encoding="utf-8"
    )
    (scripts_ops / "opsctl.sh").write_text(
        "\n".join(guard_src.REQUIRED_OPSCTL_SNIPPETS.values()), encoding="utf-8"
    )
    (scripts_ops / "storage_sqlite_local_failover.py").write_text("# contract\n", encoding="utf-8")
    (scripts_ops / "run_storage_eject_guard_launchd.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    (project_root / "scripts" / "install_storage_eject_guard_launchd.sh").write_text(
        "#!/bin/zsh\n", encoding="utf-8"
    )
    health = project_root / "governance" / "health"
    _write_json(health / "storage_failback_sync_latest.json", '{"certified_mode":"local_fallback"}\n')
    _write_json(
        health / "storage_mount_guard_latest.json",
        '{"external_available":false,"external_required_for_hot_path":true,"probe_skipped_external_io":true}\n',
    )
    _write_json(health / "ingestion_storage_control_latest.json", '{"overall_status":"ready"}\n')
    _write_json(
        health / "storage_eject_guard_latest.json",
        '{"overall_status":"ready","event":"external_available_standby","external_available":true}\n',
    )

    payload = guard_src.build_payload(project_root, check_launchd=False, check_swift_parse=False)

    assert payload["overall_status"] == "ready"
    assert payload["live_recovery"]["external_available"] is True
    assert payload["live_recovery"]["transition_event"] == "external_available_standby"
    assert "external_mount_unavailable" not in payload["live_recovery"]["blockers"]


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
        '{"overall_status":"degraded","queue_depth":50000,"write_failure_count":1,"hot_path_over_budget_bytes":4096}\n',
    )
    monkeypatch.setattr(bot_src, "_guard_payload", lambda project_root, timeout_sec: guard_src.load_json(health / "storage_reconnect_regression_guard_latest.json"))

    payload = bot_src.build_payload(project_root, apply=False, timeout_sec=90)

    names = [row["name"] for row in payload["repair_plan"]]
    assert "install_storage_eject_guard_launchd" in names
    assert "split_brain_reconcile" in names
    assert "storage_pressure_clearance" in names
    assert "global_halt_safe_refresh" in names
    assert "global_halt_safe_auto_clear" in names
    assert payload["metrics"]["repair_plan_count"] == len(names)
    split_plan = next(row for row in payload["repair_plan"] if row["name"] == "split_brain_reconcile")
    assert "--force-failback-timeout-sec" in split_plan["cmd"]
    assert split_plan["timeout_sec"] <= 75
    assert payload["metrics"]["max_repair_step_timeout_sec"] <= 90
    assert payload["metrics"]["data_plane_storage_halt_needed"] is True


def test_storage_reconnect_infrabot_ignores_non_storage_data_plane_catchup(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "storage_reconnect_regression_guard_latest.json",
        '{"overall_status":"ready","contract_ok":true,"automation":{"launchd":{"running":true,"plist_exists":true}},"live_recovery":{"split_brain_unresolved_conflicts":0,"total_pending_lines":7023}}\n',
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        '{"overall_status":"ready","backpressure":{"total_pending_lines":7023}}\n',
    )
    _write_json(health / "global_risk_killswitch_latest.json", '{"clear_blockers":[]}\n')
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        '{"overall_status":"degraded","recovery_state":"recovering_under_guard","queue_depth":4599,"write_failure_count":0,"hot_path_over_budget_bytes":0}\n',
    )
    monkeypatch.setattr(bot_src, "_guard_payload", lambda project_root, timeout_sec: guard_src.load_json(health / "storage_reconnect_regression_guard_latest.json"))

    payload = bot_src.build_payload(project_root, apply=False, timeout_sec=90)

    assert payload["overall_status"] == "ready"
    assert payload["repair_plan"] == []
    assert payload["metrics"]["data_plane_storage_halt_needed"] is False


def test_storage_reconnect_infrabot_truncates_large_child_output() -> None:
    tail = bot_src._tail_text("y" * 5000, max_chars=100)

    assert tail.startswith("...<truncated ")
    assert len(tail) < 140


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
