import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.ops import grade_regression_autopilot as grades
from scripts.ops import runtime_artifact_refresh as refresh
from scripts.ops import soak_self_healing_control as recovery


@pytest.mark.parametrize(
    "value", [None, True, -1, float("nan"), float("inf"), "100", 0]
)
def test_success_without_typed_reclamation_is_not_progress(value):
    result = {
        "ok": True,
        "parsed": {"overall_status": "deferred", "saved_bytes": value},
    }
    progress = recovery._storage_recovery_progress(
        "local_disk_cold_evidence_compaction", result
    )
    assert not progress["made_progress"]
    assert not progress["proves_local_reserve_recovered"]


@pytest.mark.parametrize(
    "name,payload",
    [
        ("local_disk_cold_evidence_compaction", {"saved_bytes": 100}),
        (
            "local_disk_lifecycle_backup_compaction",
            {"summary": {"estimated_reduction_bytes": 100}},
        ),
        (
            "local_disk_governance_telemetry_compaction",
            {"summary": {"estimated_hot_reduction_bytes": 100}},
        ),
        ("local_disk_cold_sqlite_compression", {"allocated_bytes_reclaimed": 100}),
        (
            "local_disk_resumable_deep_cold_offload",
            {"second_cold_move": {"moved_bytes": 100}},
        ),
    ],
)
def test_progress_uses_actual_owner_receipt_fields(name, payload):
    progress = recovery._storage_recovery_progress(
        name, {"ok": True, "parsed": payload}
    )
    assert progress["made_progress"]
    assert progress["reported_reclaimed_bytes"] == 100
    assert not progress["proves_local_reserve_recovered"]
    assert (
        recovery._storage_recovery_progress(name, {"ok": False, "parsed": payload})
        is None
    )


def test_empty_cleanup_backs_off_without_opening_failure_circuit(tmp_path, monkeypatch):
    now = [datetime(2026, 9, 15, tzinfo=timezone.utc)]
    monkeypatch.setattr(recovery, "_utc_now", lambda: now[0])
    calls = []
    saved = [0]

    def run(*args, **kwargs):
        calls.append(args)
        return {
            "ok": True,
            "rc": 0,
            "parsed": {"overall_status": "deferred", "saved_bytes": saved[0]},
        }

    monkeypatch.setattr(recovery, "_run_command", run)
    state = {}
    steps = []
    name = "local_disk_cold_evidence_compaction"
    options = dict(
        name=name,
        cmd=["owner", "--apply"],
        project_root=tmp_path,
        timeout_sec=30,
        env={},
        state=state,
        cooldown_seconds=60,
    )
    recovery._run_step(steps, **options)
    recovery._run_step(steps, **options)
    assert len(calls) == 1
    assert steps[-1]["executed"] is False
    for expected_delay in (120, 240, 480, 960, 1920, 3600, 3600):
        prior_until = datetime.fromisoformat(state["steps"][name]["cooldown_until_utc"])
        now[0] = prior_until + timedelta(seconds=1)
        recovery._run_step(steps, **options)
        current = state["steps"][name]
        assert (
            datetime.fromisoformat(current["cooldown_until_utc"]) - now[0]
        ).total_seconds() == expected_delay
        assert current["failure_count"] == 0
        assert not current["circuit_until_utc"]
        assert current["cooldown_reason"] == "storage_recovery_no_measured_progress"
    now[0] += timedelta(hours=2)
    saved[0] = 1000
    recovery._run_step(steps, **options)
    assert state["steps"][name]["no_progress_count"] == 0
    assert not state["steps"][name]["cooldown_until_utc"]


def test_other_successful_observers_do_not_acquire_storage_backoff(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        recovery, "_run_command", lambda *a, **k: {"ok": True, "parsed": {}}
    )
    state = {}
    row = recovery._run_step(
        [],
        name="session_ready",
        cmd=["observe"],
        project_root=tmp_path,
        timeout_sec=10,
        env={},
        state=state,
        cooldown_seconds=900,
    )
    assert "storage_recovery_progress" not in row
    assert not state["steps"]["session_ready"]["cooldown_until_utc"]


def test_lineage_scope_is_small_dependency_closed_and_serialized(tmp_path):
    originals = refresh._step_specs(tmp_path)
    specs = refresh._select_scope_specs(originals, "lineage-inputs")
    names = [row["name"] for row in specs]
    assert names == [
        "paper_replay_training",
        "runtime_training_snapshot_verified",
        "snapshot_coverage_training_verified",
        "point_in_time_event_store_verified",
        "feature_store_manifest_verified",
    ]
    assert all(set(row.get("depends_on", [])) <= set(names) for row in specs)
    assert "lineage-inputs" in refresh.SERIALIZED_PROFITABILITY_SCOPES
    assert all("--apply" not in row["cmd"] for row in specs)
    snapshot = next(
        row for row in specs if row["name"] == "runtime_training_snapshot_verified"
    )
    assert snapshot["timeout_sec"] == 125
    assert snapshot["cmd"][snapshot["cmd"].index("--max-runtime-seconds") + 1] == "120"
    original = next(
        row for row in originals if row["name"] == "runtime_training_snapshot_verified"
    )
    assert "--max-runtime-seconds" not in original["cmd"]


@pytest.mark.parametrize("nested", [False, True])
def test_grade_repairs_refresh_lineage_inputs_once_before_assessment(
    tmp_path, monkeypatch, nested
):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    (health / "training_lineage_manifest_latest.json").write_text(
        json.dumps(
            {
                "missing_contracts": [
                    "snapshot_coverage",
                    "feature_store_lineage",
                    "paper_replay_drill",
                ]
            }
        )
    )
    monkeypatch.setenv("RUNTIME_ARTIFACT_REFRESH_ACTIVE", "1" if nested else "0")
    plan = grades._repair_plan(
        tmp_path,
        {
            "surfaces": [
                {"surface": "training_quality", "state": "blocked"},
                {"surface": "training_lineage", "state": "degraded"},
            ]
        },
        storage_max_cycles=1,
    )
    inputs = [row for row in plan if "lineage-inputs" in row["cmd"]]
    assert len(inputs) == (0 if nested else 1)
    if not nested:
        assert plan[0] is inputs[0]
        assert inputs[0]["timeout_sec"] == 180
        assert (
            inputs[0]["cmd"][inputs[0]["cmd"].index("--max-run-seconds") + 1] == "165"
        )
        assert not any("paper_replay_drill.py" in " ".join(row["cmd"]) for row in plan)
    assert any("training_lineage_manifest.py" in " ".join(row["cmd"]) for row in plan)


def test_short_outer_budget_defers_epoch_instead_of_killing_snapshot(
    tmp_path, monkeypatch
):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    (health / "training_lineage_manifest_latest.json").write_text(
        json.dumps({"missing_contracts": ["snapshot_coverage"]})
    )
    monkeypatch.delenv("RUNTIME_ARTIFACT_REFRESH_ACTIVE", raising=False)
    guard = {
        "overall_status": "blocked",
        "surfaces": [{"surface": "training_quality", "state": "blocked"}],
    }
    calls = []

    def run(cmd, root, timeout):
        calls.append(cmd)
        return {"rc": 0, "payload": {"ok": False, "overall_status": "blocked"}}

    result = grades.build_payload(
        tmp_path, apply=True, timeout_sec=60, runner=run, guard_builder=lambda _: guard
    )
    assert not any("lineage-inputs" in cmd for cmd in calls)
    assert (
        result["attempts"][0]["defer_reason"]
        == "insufficient_complete_lineage_refresh_window"
    )
    assert not result["ok"]
