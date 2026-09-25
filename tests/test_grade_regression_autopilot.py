from pathlib import Path
import json

from scripts.ops import grade_regression_autopilot as src


def test_new_failed_source_triggers_refresh_even_when_lineage_summary_is_older(tmp_path):
    path = tmp_path / "governance/feature_store/latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"artifact_refresh_failed": True, "ok": False}))
    plan = src._repair_plan(tmp_path, {"surfaces": [{"surface": "training_lineage", "state": "degraded"}]}, storage_max_cycles=1)
    assert plan[0]["reason"] == "refresh_missing_lineage_inputs"
    assert "lineage-inputs" in plan[0]["cmd"]


def test_grade_regression_autopilot_runs_targeted_repairs_and_reports_final_guard() -> None:
    guard_payloads = [
        {
            "overall_status": "blocked",
            "blocked_surface_count": 2,
            "degraded_surface_count": 3,
            "surfaces": [
                {"surface": "training_lineage", "state": "degraded", "retry_budget": {"step_timeout_sec": 33, "max_attempts_per_run": 2}},
                {"surface": "storage_control", "state": "degraded", "retry_budget": {"step_timeout_sec": 44, "quiet_hours_preferred": True}},
                {"surface": "incident_closeout", "state": "degraded", "retry_budget": {"step_timeout_sec": 55}},
            ],
            "recommended_actions": ["lift training lineage", "drain storage"],
        },
        {
            "overall_status": "degraded",
            "blocked_surface_count": 0,
            "degraded_surface_count": 2,
            "surfaces": [],
            "recommended_actions": ["keep watching"],
        },
    ]
    calls: list[list[str]] = []

    def guard_builder(_: Path) -> dict:
        return guard_payloads.pop(0)

    def runner(cmd: list[str], project_root: Path, timeout_sec: int) -> dict:
        calls.append(list(cmd))
        return {"cmd": cmd, "rc": 0, "payload": {"overall_status": "ready"}, "stdout_tail": "", "stderr_tail": ""}

    payload = src.build_payload(
        Path("/tmp/project"),
        apply=True,
        runner=runner,
        guard_builder=guard_builder,
    )

    assert payload["overall_status"] == "degraded"
    assert payload["attempt_count"] >= 4
    assert any("training_lineage_manifest.py" in " ".join(cmd) for cmd in calls)
    assert any("storage_backpressure_autopilot.py" in " ".join(cmd) for cmd in calls)
    assert any("incident_closeout_autopilot.py" in " ".join(cmd) for cmd in calls)
    assert payload["regression_autopilot_contract"]["uses_per_surface_retry_budgets"] is True
    assert any(step["quiet_hours_preferred"] for step in payload["repair_plan"])
    assert payload["upgrade_track"]["upgradeable"] is True


def test_grade_regression_autopilot_omits_recursive_refresh_inside_refresh_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RUNTIME_ARTIFACT_REFRESH_ACTIVE", "1")

    plan = src._repair_plan(
        tmp_path,
        {
            "surfaces": [
                {
                    "surface": "storage_control",
                    "state": "degraded",
                    "retry_budget": {"step_timeout_sec": 30},
                }
            ]
        },
        storage_max_cycles=1,
    )

    assert not any("runtime_artifact_refresh.py" in " ".join(step.get("cmd") or []) for step in plan)
    assert any("ingestion_storage_control.py" in " ".join(step.get("cmd") or []) for step in plan)
    assert not any("storage_backpressure_autopilot.py" in " ".join(step.get("cmd") or []) for step in plan)


def test_grade_regression_autopilot_is_noop_when_every_surface_is_ready(tmp_path: Path) -> None:
    plan = src._repair_plan(
        tmp_path,
        {
            "overall_status": "ready",
            "surfaces": [
                {"surface": "training_quality", "state": "ready"},
                {"surface": "storage_control", "state": "ready"},
            ],
        },
        storage_max_cycles=1,
    )

    assert plan == []


def test_grade_regression_autopilot_never_embeds_full_artifact_refresh(tmp_path: Path) -> None:
    plan = src._repair_plan(
        tmp_path,
        {
            "overall_status": "blocked",
            "surfaces": [
                {"surface": "training_quality", "state": "blocked"},
                {"surface": "storage_control", "state": "degraded"},
            ],
        },
        storage_max_cycles=1,
    )

    assert plan
    assert not any("runtime_artifact_refresh.py" in " ".join(step.get("cmd") or []) for step in plan)


def test_repairs_share_work_deadline_and_preserve_deferred_debt(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(src.time, "monotonic", lambda: now[0])
    guard = {
        "overall_status": "degraded",
        "surfaces": [{"surface": "incident_closeout", "state": "degraded"}],
    }
    calls = []

    def run(cmd, root, timeout):
        calls.append((cmd, timeout))
        now[0] += timeout
        return {"cmd": cmd, "rc": 124, "timeout_cleanup": {"reaped": True}}

    payload = src.build_payload(
        tmp_path,
        apply=True,
        timeout_sec=20,
        runner=run,
        guard_builder=lambda root: guard,
    )
    assert len(calls) == 1
    assert calls[0][1] == 20 - src.CLEANUP_RESERVE_SECONDS
    assert payload["deferred_attempt_count"] == 3
    assert not payload["ok"]
    assert payload["attempts"][0]["timeout_cleanup"]["reaped"]
    assert all(
        row["defer_reason"] == "repair_cycle_deadline"
        for row in payload["attempts"][1:]
    )


def test_repair_plan_refreshes_dependencies_before_consumers_without_pdf(tmp_path):
    plan = src._repair_plan(
        tmp_path,
        {
            "surfaces": [
                {"surface": "training_lineage", "state": "degraded"},
                {"surface": "promotion_autopilot", "state": "degraded"},
                {"surface": "incident_closeout", "state": "degraded"},
            ]
        },
        storage_max_cycles=1,
    )
    scripts = [Path(step["cmd"][1]).name for step in plan]
    assert scripts.count("promotion_packet_builder.py") == 1
    assert scripts.count("promotion_autopilot_packet.py") == 1
    assert scripts.count("retrain_schema_compatibility_guard.py") == 1
    assert scripts.index("retrain_schema_compatibility_guard.py") < scripts.index("promotion_packet_builder.py")
    assert scripts.index("promotion_packet_builder.py") < scripts.index("promotion_autopilot_packet.py")
    builder = next(step for step in plan if "promotion_packet_builder.py" in step["cmd"][1])
    assert builder["cmd"][2:] == ["--json"]
    assert scripts.index("promotion_autopilot_packet.py") < scripts.index(
        "training_lineage_manifest.py"
    )
    assert (
        scripts.index("incident_timeline.py")
        < scripts.index("data_plane_recovery_controller.py")
        < scripts.index("incident_review_packet.py")
        < scripts.index("incident_closeout_autopilot.py")
    )
    review = next(
        step for step in plan if "incident_review_packet.py" in step["cmd"][1]
    )
    assert "--no-render-pdf" in review["cmd"]


def test_training_quality_is_reassessed_after_lineage_not_against_stale_manifest(tmp_path):
    plan = src._repair_plan(tmp_path, {"surfaces": [
        {"surface": "training_quality", "state": "blocked"},
        {"surface": "training_lineage", "state": "degraded"},
        {"surface": "autonomy_control", "state": "degraded"},
    ]}, storage_max_cycles=1)
    scripts = [Path(step["cmd"][1]).name for step in plan]
    assert scripts.index("training_lineage_manifest.py") < scripts.index("training_quality_control.py") < scripts.index("autonomy_control_plane.py")
    assert scripts.count("training_quality_control.py") == 1


def test_missing_restore_receipt_uses_bounded_native_recovery_before_consumers(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNTIME_ARTIFACT_REFRESH_ACTIVE", raising=False)
    plan = src._repair_plan(tmp_path, {"surfaces": [{
        "surface": "storage_control", "state": "blocked",
        "retry_budget": {"step_timeout_sec": 900, "quiet_hours_preferred": True},
    }]}, storage_max_cycles=1)
    assert plan[0]["cmd"] == [str(tmp_path / "scripts/ops/opsctl.sh"),
                              "state-snapshot-drill", "--recover-latest-verified", "--json"]
    assert plan[0]["timeout_sec"] == 200
    assert plan[0]["quiet_hours_preferred"]
    assert "storage_resilience_control.py" in plan[1]["cmd"][1]
    assert "--fast" in plan[1]["cmd"]
    assert "ingestion_storage_control.py" in plan[2]["cmd"][1]
    monkeypatch.setenv("RUNTIME_ARTIFACT_REFRESH_ACTIVE", "1")
    nested = src._repair_plan(tmp_path, {"surfaces": [{"surface": "storage_control", "state": "blocked"}]}, storage_max_cycles=1)
    assert not any("--recover-latest-verified" in step["cmd"] for step in nested)


def test_complete_restore_receipt_does_not_trigger_recovery(tmp_path):
    path = tmp_path / "exports/state_snapshot_drills/latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"ok": True, "files_checked": 5,
                               "files_restore_verified": 5, "latest_write_verified": True,
                               "published_latest_write_verified": True}))
    assert not src._restore_receipt_recovery_needed(tmp_path)
    path.write_text("invalid")
    assert src._restore_receipt_recovery_needed(tmp_path)


def test_owner_resource_deferral_remains_visible(tmp_path):
    guard = {"overall_status": "blocked", "surfaces": [{"surface": "training_quality", "state": "blocked"}]}
    payload = src.build_payload(tmp_path, apply=True, guard_builder=lambda _: guard,
        runner=lambda cmd, root, timeout: {"cmd": cmd, "rc": 2, "payload": {
            "overall_status": "deferred", "reason": "resource_hold"}})
    assert payload["deferred_attempt_count"] == 1
    assert payload["attempts"][0]["defer_reason"] == "resource_hold"
    assert not payload["ok"]


def test_lineage_repairs_actual_replay_dependency_before_signed_packet(tmp_path):
    path = tmp_path / "governance/health/training_lineage_manifest_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"missing_contracts": ["paper_replay_drill"]}))
    plan = src._repair_plan(
        tmp_path,
        {
            "surfaces": [
                {"surface": "autonomy_control", "state": "degraded"},
                {"surface": "training_lineage", "state": "degraded"},
                {"surface": "promotion_autopilot", "state": "degraded"},
            ]
        },
        storage_max_cycles=1,
    )
    scripts = [Path(row["cmd"][1]).name for row in plan]
    expected = [
        "paper_replay_drill.py",
        "retrain_schema_compatibility_guard.py",
        "walk_forward_validate.py",
        "walk_forward_promotion_gate.py",
        "promotion_readiness_summary.py",
        "promotion_packet_builder.py",
        "promotion_autopilot_packet.py",
        "training_lineage_manifest.py",
        "coverage_gap_closer.py",
        "incident_timeline.py",
        "runtime_throttle_control.py",
        "autonomy_control_plane.py",
    ]
    assert scripts == expected
    replay = plan[0]
    assert replay["cmd"][2:5] == ["--hours", "336", "--strict-exit"]
    assert "--min-rows" not in replay["cmd"]
    assert replay["timeout_sec"] == 60
    coverage = next(step for step in plan if Path(step["cmd"][1]).name == "coverage_gap_closer.py")
    assert coverage["cmd"][2:] == ["--skip-refresh", "--json"]
    assert coverage["timeout_sec"] == 30
    assert not any(
        flag in row["cmd"]
        for row in plan
        for flag in (
            "--launch",
            "--apply-stage",
            "--auto-launch-off-hours",
            "--run-master-update",
        )
    )


def test_noop_replay_repair_does_not_clear_evidence_debt(tmp_path):
    path = tmp_path / "governance/health/training_lineage_manifest_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"missing_contracts": ["paper_replay_drill"]}))
    guard = {
        "overall_status": "degraded",
        "degraded_surface_count": 1,
        "surfaces": [{"surface": "training_lineage", "state": "degraded"}],
    }
    result = src.build_payload(
        tmp_path,
        apply=True,
        guard_builder=lambda _: guard,
        runner=lambda cmd, root, timeout: {
            "cmd": cmd,
            "rc": 2,
            "payload": {"ok": False, "failed_checks": ["paper_rows_low"]},
        },
    )
    assert result["final_guard"]["degraded_surface_count"] == 1
    assert result["ok"] is False
    assert result["attempts"][0]["rc"] == 2


def test_json_diagnostics_have_character_caps_without_losing_payload(
    tmp_path, monkeypatch
):
    raw = {"ok": False, "nested": "x" * 30000}
    monkeypatch.setattr(
        src,
        "run_bounded_process_group",
        lambda *a, **kw: {
            "rc": 2,
            "stdout": json.dumps(raw),
            "stderr": "y" * 12000,
        },
    )
    result = src._run(["observer"], tmp_path, 10)
    assert result["payload"] == raw
    assert len(result["stdout_tail"]) == 4000
    assert len(result["stderr_tail"]) == 4000
