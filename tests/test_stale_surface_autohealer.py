import json
from pathlib import Path


from scripts.ops import infrastructure_autofix_bot as infra_src
from scripts.ops import stale_surface_autohealer as healer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_stale_surface_autohealer_plans_allowlisted_repairs(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "artifact_freshness_slo_latest.json",
        {
            "overall_status": "blocked",
            "artifacts": [
                {
                    "name": "session_ready",
                    "stale": True,
                    "required": True,
                    "age_minutes": 20,
                    "max_age_minutes": 15,
                    "refresh_command": "./scripts/session_ready_check.py --json",
                },
                {
                    "name": "unsafe_surface",
                    "stale": True,
                    "required": False,
                    "refresh_command": "rm -rf /",
                },
            ],
        },
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "watchdog_intelligence": {
                "exact_needs": [
                    {
                        "target": "all_sleeves",
                        "status": "needs_repair",
                        "blocker": "heartbeat_stale",
                        "exact_command": ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--json"],
                    },
                    {
                        "target": "coinbase_loop",
                        "status": "intentional_hold",
                        "blocker": "operator_mode",
                        "exact_command": ["./scripts/ops/opsctl.sh", "coinbase-api-health", "--json"],
                    },
                ]
            }
        },
    )

    monkeypatch.setattr(
        healer,
        "_run_text",
        lambda cmd, *, cwd, timeout_sec=20: {
            "rc": 0,
            "stdout": "PID\tStatus\tLabel\n-\t0\tcom.dankingsley.schwab.codex.trainingdone.pcore.20260627_164407\n",
            "stderr": "",
            "stdout_tail": "",
            "stderr_tail": "",
        },
    )

    payload = healer.build_payload(project_root, apply=False, max_artifact_repairs=4, max_process_repairs=4)

    names = [row["name"] for row in payload["repair_plan"]]
    assert "session_ready" in names
    assert "all_sleeves" in names
    assert "unsafe_surface" in names
    assert "coinbase_loop" not in names
    assert any(row["surface"] == "stale_one_shot_launchd" for row in payload["repair_plan"])
    assert payload["metrics"]["manual_review_count"] == 1
    assert payload["overall_status"] == "degraded"


def test_stale_surface_autohealer_apply_removes_stale_completed_notice(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "artifact_freshness_slo_latest.json", {"overall_status": "ready", "artifacts": []})
    _write_json(health / "process_watchdog_latest.json", {"watchdog_intelligence": {"exact_needs": []}})
    _write_json(
        health / "codex_training_done_notice_latest.json",
        {
            "state": "completed",
            "final_status": "completed_successfully",
            "ended_utc": "2026-01-01T00:00:00+00:00",
        },
    )
    txt_path = health / "codex_training_done_notice_latest.txt"
    txt_path.write_text("Training done: status=completed_successfully\n", encoding="utf-8")

    monkeypatch.setattr(
        healer,
        "_run_text",
        lambda cmd, *, cwd, timeout_sec=20: {
            "rc": 0,
            "stdout": "PID\tStatus\tLabel\n",
            "stderr": "",
            "stdout_tail": "",
            "stderr_tail": "",
        },
    )

    payload = healer.build_payload(
        project_root,
        apply=True,
        refresh_inputs=False,
        max_completed_notice_age_minutes=0,
    )

    assert payload["metrics"]["file_cleanup_count"] == 2
    assert all(row["rc"] == 0 for row in payload["attempts"])
    assert not (health / "codex_training_done_notice_latest.json").exists()
    assert not txt_path.exists()


def test_stale_surface_autohealer_optional_artifact_timeout_is_degraded_not_blocked(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "artifact_freshness_slo_latest.json",
        {
            "overall_status": "degraded",
            "artifacts": [
                {
                    "name": "sentiment_report",
                    "stale": True,
                    "required": False,
                    "refresh_command": "./scripts/ops/opsctl.sh sentiment-report --json",
                }
            ],
        },
    )
    _write_json(health / "process_watchdog_latest.json", {"watchdog_intelligence": {"exact_needs": []}})

    monkeypatch.setattr(
        healer,
        "_run_text",
        lambda cmd, *, cwd, timeout_sec=20: {
            "rc": 0,
            "stdout": "PID\tStatus\tLabel\n",
            "stderr": "",
            "stdout_tail": "",
            "stderr_tail": "",
        },
    )
    monkeypatch.setattr(
        healer,
        "_run_command",
        lambda cmd, *, cwd, timeout_sec: {
            "cmd": cmd,
            "rc": 124,
            "timed_out": True,
            "duration_ms": 1.0,
            "stdout_tail": "",
            "stderr_tail": "timeout",
        },
    )

    payload = healer.build_payload(project_root, apply=True, refresh_inputs=False)

    assert payload["overall_status"] == "degraded"
    assert payload["failed_attempts"][0]["required"] is False
    assert payload["hard_failed_attempts"] == []


def test_infrastructure_autofix_routes_stale_signals_to_autohealer(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "artifact_freshness_slo_latest.json",
        {"overall_status": "ready", "sla_summary": {"stale_required": 0, "stale_optional": 1}},
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {"overall_status": "ready", "watchdog_intelligence": {"active_issue_count": 0}},
    )
    _write_json(
        health / "remote_alert_control_latest.json",
        {"overall_status": "ready", "channels": {"any_configured": True}, "critical_backlog": {"unsent_count": 0}},
    )

    payload = infra_src.build_payload(project_root, apply=False)

    names = [row["name"] for row in payload["repair_plan"]]
    assert "stale_surface_autohealer" in names
    assert payload["metrics"]["artifact_freshness_stale_optional"] == 1
