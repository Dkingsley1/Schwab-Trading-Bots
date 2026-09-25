import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import ops_scheduled_job_catalog as catalog  # noqa: E402
from scripts.ops import ops_scheduled_job_lifecycle as src  # noqa: E402


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _touch_runner(project_root: Path, spec: catalog.JobSpec) -> None:
    _write_text(project_root / spec.runner, "#!/bin/zsh\n")


def _touch_plist(
    launch_dir: Path, spec: catalog.JobSpec, *, wrapped: bool = True
) -> None:
    runner = "<string>run_scheduled_lifecycle_job.py</string>" if wrapped else ""
    _write_text(launch_dir / f"{spec.label}.plist", f"<plist>{runner}</plist>\n")


def _installer_text(
    *, active: list[catalog.JobSpec], removed: list[catalog.JobSpec]
) -> str:
    lines = [f'install_job "{spec.label}"' for spec in active]
    lines.extend(f'remove_job "{spec.label}"' for spec in removed)
    return "\n".join(lines) + "\n"


def _job(job_id: str, artifact: str, *, removed: bool = False) -> catalog.JobSpec:
    return catalog._job(
        job_id,
        "governance",
        f"scripts/ops/{job_id}.py",
        artifact,
        cadence_seconds=300,
        resource_class="governance_guard",
        deadline_seconds=120,
        owner="test_owner",
        authority_boundary="read_only_test_boundary",
        install_policy=(
            catalog.REMOVED_INSTALL_POLICY if removed else catalog.ACTIVE_INSTALL_POLICY
        ),
    )


def _artifact(now: datetime, lifecycle: dict | None) -> dict:
    payload = {"timestamp_utc": now.isoformat(), "overall_status": "ready"}
    if lifecycle is not None:
        payload["job_lifecycle"] = lifecycle
    return payload


def _lifecycle(
    run_id: str,
    *,
    deferred: bool = False,
    failed: bool = False,
    next_eligible_utc: str = "2026-09-09T12:05:00+00:00",
) -> dict:
    return {
        "run_id": run_id,
        "scheduled": True,
        "eligible": not deferred,
        "deferred": deferred,
        "deferred_reason": "quiet_window" if deferred else "",
        "started": not deferred,
        "completed": True,
        "failed": failed,
        "next_eligible_utc": next_eligible_utc,
    }


def test_scheduled_job_lifecycle_groups_ready_deferred_and_lifecycle_debt(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    project_root = tmp_path / "project"
    launch_dir = tmp_path / "LaunchAgents"
    ready = _job("ready_job", "governance/health/ready_latest.json")
    deferred = _job("deferred_job", "governance/health/deferred_latest.json")
    debt = _job("debt_job", "governance/health/debt_latest.json")
    removed = _job(
        "removed_job",
        "governance/health/removed_latest.json",
        removed=True,
    )
    jobs = (ready, deferred, debt, removed)

    _write_text(
        project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh",
        _installer_text(active=[ready, deferred, debt], removed=[removed]),
    )
    for spec in jobs:
        _touch_runner(project_root, spec)
    for spec in (ready, deferred, debt):
        _touch_plist(launch_dir, spec)
    _write_json(
        project_root / ready.artifact,
        _artifact(now, _lifecycle("ready-run")),
    )
    _write_json(
        project_root / deferred.artifact,
        _artifact(now, _lifecycle("deferred-run", deferred=True)),
    )
    _write_json(project_root / debt.artifact, _artifact(now, None))

    payload = src.build_payload(
        project_root,
        launch_agents_dir=launch_dir,
        jobs=jobs,
        now=now,
    )

    rows = {row["job_id"]: row for row in payload["jobs"]}
    assert payload["overall_status"] == "ready_with_lifecycle_debt"
    assert payload["ok"] is True
    assert payload["hard_issue_count"] == 0
    assert payload["deferred_job_count"] == 1
    assert payload["lifecycle_debt_count"] == 1
    assert payload["action_queue_count"] == 1
    assert payload["installer_alignment"]["ok"] is True
    assert rows["ready_job"]["operational_status"] == "ready"
    assert rows["deferred_job"]["operational_status"] == "deferred"
    assert rows["debt_job"]["operational_status"] == "lifecycle_debt"
    assert rows["debt_job"]["issues"] == ["lifecycle_receipt_missing"]
    assert rows["removed_job"]["operational_status"] == "retired"
    assert payload["action_queue"][0]["action_id"] == "close_lifecycle_receipt_debt_job"


def test_scheduled_job_lifecycle_treats_fresh_deferral_as_managed_liveness(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    project_root = tmp_path / "project"
    launch_dir = tmp_path / "LaunchAgents"
    deferred = _job("deferred_job", "governance/health/deferred_latest.json")

    _write_text(
        project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh",
        _installer_text(active=[deferred], removed=[]),
    )
    _touch_runner(project_root, deferred)
    _touch_plist(launch_dir, deferred)
    lifecycle = _lifecycle("deferred-run", deferred=True)
    lifecycle["completed_utc"] = (now - timedelta(seconds=30)).isoformat()
    lifecycle["started_utc"] = (now - timedelta(seconds=31)).isoformat()
    _write_json(
        project_root / deferred.artifact,
        {
            "timestamp_utc": (now - timedelta(minutes=20)).isoformat(),
            "overall_status": "ready",
            "job_lifecycle": lifecycle,
        },
    )

    payload = src.build_payload(
        project_root,
        launch_agents_dir=launch_dir,
        jobs=(deferred,),
        now=now,
    )

    row = payload["jobs"][0]
    assert payload["overall_status"] == "ready_with_deferrals"
    assert payload["ok"] is True
    assert payload["hard_issue_count"] == 0
    assert payload["evidence_stale_count"] == 0
    assert payload["managed_stale_deferral_count"] == 1
    assert row["operational_status"] == "deferred"
    assert row["stale_evidence_managed_by_fresh_deferral"] is True
    assert row["issues"] == ["evidence_stale_under_fresh_deferral"]


def test_scheduled_job_lifecycle_keeps_old_deferral_stale_evidence_hard(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    project_root = tmp_path / "project"
    launch_dir = tmp_path / "LaunchAgents"
    deferred = _job("deferred_job", "governance/health/deferred_latest.json")

    _write_text(
        project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh",
        _installer_text(active=[deferred], removed=[]),
    )
    _touch_runner(project_root, deferred)
    _touch_plist(launch_dir, deferred)
    lifecycle = _lifecycle("old-deferred-run", deferred=True)
    lifecycle["completed_utc"] = (now - timedelta(hours=2)).isoformat()
    lifecycle["started_utc"] = (now - timedelta(hours=2, seconds=1)).isoformat()
    _write_json(
        project_root / deferred.artifact,
        {
            "timestamp_utc": (now - timedelta(minutes=20)).isoformat(),
            "overall_status": "ready",
            "job_lifecycle": lifecycle,
        },
    )

    payload = src.build_payload(
        project_root,
        launch_agents_dir=launch_dir,
        jobs=(deferred,),
        now=now,
    )

    row = payload["jobs"][0]
    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert payload["hard_issue_count"] == 1
    assert payload["evidence_stale_count"] == 1
    assert payload["managed_stale_deferral_count"] == 0
    assert row["operational_status"] == "evidence_stale"
    assert row["stale_evidence_managed_by_fresh_deferral"] is False
    assert row["issues"] == ["evidence_stale"]


def test_scheduled_job_lifecycle_accepts_fresh_persistent_writer_without_wrapper_receipt(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    project_root = tmp_path / "project"
    launch_dir = tmp_path / "LaunchAgents"
    writer = catalog._job(
        "sql_link_writer",
        "storage",
        "scripts/ops/run_sql_link_writer_launchd.sh",
        "governance/health/sql_link_service_latest.json",
        cadence_seconds=180,
        resource_class="single_writer",
        deadline_seconds=0,
        owner="sql_link_writer",
        authority_boundary="single_sqlite_writer_no_live_orders",
    )

    _write_text(
        project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh",
        _installer_text(active=[writer], removed=[]),
    )
    _touch_runner(project_root, writer)
    _touch_plist(launch_dir, writer, wrapped=False)
    _write_json(
        project_root / writer.artifact,
        {
            "timestamp_utc": (now - timedelta(seconds=30)).isoformat(),
            "ok": True,
            "overall_status": "ready",
        },
    )

    payload = src.build_payload(
        project_root,
        launch_agents_dir=launch_dir,
        jobs=(writer,),
        now=now,
    )

    row = payload["jobs"][0]
    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["lifecycle_debt_count"] == 0
    assert row["operational_status"] == "ready"
    assert row["single_writer_fresh_without_receipt"] is True
    assert row["issues"] == ["single_writer_fresh_artifact_liveness"]
    assert row["lifecycle_wrapper_required"] is False


def test_bounded_single_writer_cannot_mask_missing_receipt_with_fresh_data(tmp_path):
    now = datetime.now(timezone.utc)
    writer = catalog._job(
        "sql_link_writer",
        "storage",
        "scripts/ops/run_sql_link_writer_launchd.sh",
        "governance/health/sql_link_service_latest.json",
        cadence_seconds=60,
        resource_class="single_writer",
        deadline_seconds=900,
        owner="sql_link_writer",
        authority_boundary="single_sqlite_writer_no_live_orders",
    )
    project_root = tmp_path / "project"
    launch_dir = tmp_path / "LaunchAgents"
    _touch_runner(project_root, writer)
    _touch_plist(launch_dir, writer, wrapped=False)
    _write_json(
        project_root / writer.artifact, {"timestamp_utc": now.isoformat(), "ok": True}
    )
    row = src._job_row(project_root, launch_dir, writer, now=now)
    assert row["lifecycle_wrapper_required"] is True
    assert row["single_writer_fresh_without_receipt"] is False
    assert "lifecycle_wrapper_missing" in row["issues"]
    assert "lifecycle_receipt_missing" in row["issues"]


def test_scheduled_job_lifecycle_blocks_missing_installation_and_unexpected_legacy(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    project_root = tmp_path / "project"
    launch_dir = tmp_path / "LaunchAgents"
    active = _job("active_job", "governance/health/active_latest.json")
    removed = _job(
        "legacy_job",
        "governance/health/legacy_latest.json",
        removed=True,
    )
    jobs = (active, removed)

    _write_text(
        project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh",
        _installer_text(active=[active], removed=[removed]),
    )
    _touch_runner(project_root, active)
    _touch_plist(launch_dir, removed)
    _write_json(
        project_root / active.artifact,
        _artifact(now, _lifecycle("active-run")),
    )

    payload = src.build_payload(
        project_root,
        launch_agents_dir=launch_dir,
        jobs=jobs,
        now=now,
    )

    rows = {row["job_id"]: row for row in payload["jobs"]}
    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert payload["hard_issue_count"] == 2
    assert payload["installation_missing_count"] == 1
    assert payload["unexpected_legacy_installed_count"] == 1
    assert payload["installer_alignment"]["ok"] is True
    assert rows["active_job"]["operational_status"] == "installation_missing"
    assert rows["legacy_job"]["operational_status"] == "unexpected_legacy_installed"
    assert rows["legacy_job"]["lifecycle_wrapper_required"] is False


def test_scheduled_job_lifecycle_action_queue_flags_unwrapped_plist(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    project_root = tmp_path / "project"
    launch_dir = tmp_path / "LaunchAgents"
    active = _job("active_job", "governance/health/active_latest.json")

    _write_text(
        project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh",
        _installer_text(active=[active], removed=[]),
    )
    _touch_runner(project_root, active)
    _touch_plist(launch_dir, active, wrapped=False)
    _write_json(
        project_root / active.artifact,
        _artifact(now, _lifecycle("active-run")),
    )

    payload = src.build_payload(
        project_root,
        launch_agents_dir=launch_dir,
        jobs=(active,),
        now=now,
    )

    row = payload["jobs"][0]
    assert row["operational_status"] == "lifecycle_debt"
    assert row["lifecycle_wrapper_present"] is False
    assert "lifecycle_wrapper_missing" in row["issues"]
    assert payload["lifecycle_wrapper_missing_count"] == 1
    assert (
        payload["action_queue"][0]["action_id"] == "reinstall_lifecycle_wrapped_launchd"
    )
    assert (
        payload["action_queue"][0]["command"]
        == "./scripts/ops/install_ops_automation_launchd.sh"
    )
    assert (
        payload["action_queue"][0]["preflight"]["status"]
        == "ready_for_operator_confirmation"
    )
    assert payload["action_queue"][0]["preflight"]["command_allowlisted"] is True
    assert payload["action_queue_preflight"]["safe_to_review"] is True


def test_scheduled_job_lifecycle_preflight_blocks_receipt_until_wrapper_installed(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    project_root = tmp_path / "project"
    launch_dir = tmp_path / "LaunchAgents"
    active = _job("active_job", "governance/health/active_latest.json")

    _write_text(
        project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh",
        _installer_text(active=[active], removed=[]),
    )
    _touch_runner(project_root, active)
    _touch_plist(launch_dir, active, wrapped=False)
    _write_json(project_root / active.artifact, _artifact(now, None))

    payload = src.build_payload(
        project_root,
        launch_agents_dir=launch_dir,
        jobs=(active,),
        now=now,
    )

    actions = {action["action_id"]: action for action in payload["action_queue"]}
    receipt_preflight = actions["close_lifecycle_receipt_active_job"]["preflight"]
    assert receipt_preflight["status"] == "blocked_by_prerequisite"
    assert "lifecycle_wrapper_not_installed" in receipt_preflight["blocked_by"]
    assert (
        actions["reinstall_lifecycle_wrapped_launchd"]["preflight"]["status"]
        == "ready_for_operator_confirmation"
    )
    assert payload["action_queue_preflight"]["blocked_action_count"] == 1


def test_scheduled_job_lifecycle_preflight_rejects_unknown_command() -> None:
    result = src._action_preflight(
        {
            "action_id": "unsafe",
            "command": "rm -rf /tmp/example",
            "auto_execute": False,
        },
        rows=[],
        known_labels=set(),
    )

    assert result["status"] == "blocked_by_prerequisite"
    assert result["command_family"] == "unknown"
    assert "command_not_allowlisted" in result["blocked_by"]


def test_scheduled_job_lifecycle_cli_applies_catalog_artifact(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    out_file = tmp_path / "ops_scheduled_job_lifecycle_latest.json"

    def _fake_build_payload(
        project_root: Path, *, launch_agents_dir: Path | None = None
    ) -> dict:
        return {
            "schema_version": src.SCHEMA_VERSION,
            "timestamp_utc": "2026-09-09T12:00:00+00:00",
            "source": "ops_scheduled_job_lifecycle",
            "overall_status": "ready",
            "ok": True,
            "project_root": str(project_root),
            "launch_agents_dir": str(launch_agents_dir),
            "job_count": 0,
            "hard_issue_count": 0,
            "lifecycle_debt_count": 0,
        }

    monkeypatch.setattr(src, "build_payload", _fake_build_payload)

    rc = src.main(
        [
            "--project-root",
            str(tmp_path / "project"),
            "--launch-agents-dir",
            str(tmp_path / "LaunchAgents"),
            "--out-file",
            str(out_file),
            "--apply",
            "--json",
        ]
    )

    assert rc == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert stdout_payload == written
    assert written["overall_status"] == "ready"


def test_scheduled_job_lifecycle_cli_queue_only_view(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def _fake_build_payload(
        project_root: Path, *, launch_agents_dir: Path | None = None
    ) -> dict:
        return {
            "schema_version": src.SCHEMA_VERSION,
            "timestamp_utc": "2026-09-09T12:00:00+00:00",
            "source": "ops_scheduled_job_lifecycle",
            "overall_status": "blocked",
            "ok": False,
            "job_count": 2,
            "active_job_count": 2,
            "hard_issue_count": 1,
            "lifecycle_debt_count": 1,
            "lifecycle_wrapper_missing_count": 1,
            "evidence_missing_count": 1,
            "evidence_stale_count": 0,
            "action_queue_count": 1,
            "action_queue": [
                {
                    "priority": "P1",
                    "action_id": "refresh_missing_artifact_runtime_smooth_mode",
                }
            ],
            "jobs": [{"job_id": "runtime_smooth_mode"}],
            "recommended_actions": ["repair missing artifact"],
        }

    monkeypatch.setattr(src, "build_payload", _fake_build_payload)

    rc = src.main(
        ["--project-root", str(tmp_path / "project"), "--queue-only", "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert "jobs" not in payload
    assert payload["action_queue_count"] == 1
    assert (
        payload["action_queue"][0]["action_id"]
        == "refresh_missing_artifact_runtime_smooth_mode"
    )


def test_scheduled_job_lifecycle_cli_preflight_only_view(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def _fake_build_payload(
        project_root: Path, *, launch_agents_dir: Path | None = None
    ) -> dict:
        return {
            "schema_version": src.SCHEMA_VERSION,
            "timestamp_utc": "2026-09-09T12:00:00+00:00",
            "source": "ops_scheduled_job_lifecycle",
            "overall_status": "blocked",
            "ok": False,
            "job_count": 2,
            "active_job_count": 2,
            "hard_issue_count": 1,
            "lifecycle_debt_count": 1,
            "action_queue_count": 1,
            "action_queue_preflight": {
                "safe_to_review": True,
                "ready_for_operator_confirmation_count": 1,
            },
            "action_queue": [
                {
                    "priority": "P1",
                    "action_id": "reinstall_lifecycle_wrapped_launchd",
                    "category": "lifecycle_adoption",
                    "job_id": "",
                    "issue": "lifecycle_wrapper_missing",
                    "command": "./scripts/ops/install_ops_automation_launchd.sh",
                    "preflight": {"status": "ready_for_operator_confirmation"},
                }
            ],
            "jobs": [{"job_id": "active_job"}],
            "recommended_actions": ["reinstall wrappers"],
        }

    monkeypatch.setattr(src, "build_payload", _fake_build_payload)

    rc = src.main(
        ["--project-root", str(tmp_path / "project"), "--preflight-only", "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert "jobs" not in payload
    assert "action_queue" not in payload
    assert payload["actions"][0]["action_id"] == "reinstall_lifecycle_wrapped_launchd"
    assert (
        payload["actions"][0]["preflight"]["status"]
        == "ready_for_operator_confirmation"
    )
