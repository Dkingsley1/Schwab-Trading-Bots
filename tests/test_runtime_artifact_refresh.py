import json
import os
from contextlib import contextmanager
from pathlib import Path

from scripts.ops import runtime_artifact_refresh


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_runtime_artifact_refresh_caps_single_line_diagnostic_tails() -> None:
    raw = "prefix-" + ("x" * 10_000)

    tail = runtime_artifact_refresh._tail_text(raw, max_chars=200)

    assert tail.startswith("...[truncated ")
    assert tail.endswith("x" * 200)
    assert len(tail) < 260


def test_runtime_artifact_refresh_parses_pretty_printed_json_output() -> None:
    payload = {"overall_status": "ready", "nested": {"value": 3}}

    assert runtime_artifact_refresh._parse_json_output(json.dumps(payload, indent=2)) == payload


def test_runtime_artifact_refresh_understands_nested_dashboard_status() -> None:
    payload = {"overall": {"status": "degraded", "ok": False}}

    status = runtime_artifact_refresh._step_status({"rc": 2, "payload": payload})
    summary = runtime_artifact_refresh._payload_summary(payload)

    assert status == "degraded"
    assert summary == {"overall_status": "degraded", "ok": False}


def test_runtime_artifact_refresh_marks_child_maintenance_context(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_run(cmd, *, cwd, timeout_seconds, env):
        captured.update({"cmd": cmd, "cwd": cwd, "timeout_seconds": timeout_seconds, "env": env})
        return {
            "rc": 0,
            "stdout": json.dumps({"overall_status": "ready", "ok": True}),
            "stderr": "",
            "timed_out": False,
            "timeout_cleanup": {"reaped": True},
        }

    monkeypatch.setattr(runtime_artifact_refresh, "run_bounded_process_group", fake_run)
    result = runtime_artifact_refresh._run_spec(
        {
            "cmd": ["producer", "--json"],
            "payload_path": tmp_path / "producer_latest.json",
            "timeout_sec": 17,
        },
        tmp_path,
    )

    assert result["rc"] == 0
    assert captured["env"][runtime_artifact_refresh.REFRESH_ACTIVE_ENV] == "1"


def test_runtime_artifact_refresh_skips_nested_entry_without_overwriting_outer_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "runtime_artifact_refresh_latest.json"
    monkeypatch.setenv(runtime_artifact_refresh.REFRESH_ACTIVE_ENV, "1")
    monkeypatch.setattr(
        runtime_artifact_refresh.sys,
        "argv",
        ["runtime_artifact_refresh.py", "--out-file", str(out_path), "--json"],
    )

    assert runtime_artifact_refresh.main() == 0
    assert not out_path.exists()


def test_profitability_scope_holds_generation_lock_for_the_whole_epoch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_lock(project_root: Path, *, timeout_seconds: float):
        assert project_root == tmp_path
        assert timeout_seconds == 120.0
        events.append("acquired")
        yield object()
        events.append("released")

    def fake_build(project_root: Path, *, scope: str):
        assert project_root == tmp_path
        assert scope == "training-profitability"
        assert os.environ[runtime_artifact_refresh.PAPER_PROFITABILITY_LOCK_ENV] == "1"
        events.append("built")
        return {"ok": True, "overall_status": "ready"}

    monkeypatch.delenv(runtime_artifact_refresh.PAPER_PROFITABILITY_LOCK_ENV, raising=False)
    monkeypatch.setattr(runtime_artifact_refresh, "paper_profitability_generation_lock", fake_lock)
    monkeypatch.setattr(runtime_artifact_refresh, "build_payload", fake_build)

    payload = runtime_artifact_refresh.build_payload_serialized(
        tmp_path,
        scope="training-profitability",
    )

    assert events == ["acquired", "built", "released"]
    assert runtime_artifact_refresh.PAPER_PROFITABILITY_LOCK_ENV not in os.environ
    assert payload["single_writer_epoch_lock"]["held"] is True
    assert payload["single_writer_epoch_lock"]["prevents_interleaved_mutable_latest_publication"] is True


def test_runtime_artifact_refresh_reports_recovered_and_blocked_outputs(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    ready_path = health / "ready_latest.json"
    blocked_path = health / "blocked_latest.json"
    missing_path = health / "missing_latest.json"

    specs = [
        {"name": "ready_artifact", "payload_path": ready_path, "cmd": ["ready"]},
        {"name": "blocked_artifact", "payload_path": blocked_path, "cmd": ["blocked"]},
        {"name": "missing_artifact", "payload_path": missing_path, "cmd": ["missing"]},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        name = spec["name"]
        if name == "ready_artifact":
            _write_json(Path(spec["payload_path"]), {"timestamp_utc": "2026-04-22T16:00:00Z", "ok": True, "overall_status": "ready"})
            return {"cmd": list(spec["cmd"]), "rc": 0, "payload": {"ok": True, "overall_status": "ready"}, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}
        if name == "blocked_artifact":
            _write_json(Path(spec["payload_path"]), {"timestamp_utc": "2026-04-22T16:00:00Z", "ok": False, "overall_status": "blocked"})
            return {"cmd": list(spec["cmd"]), "rc": 2, "payload": {"ok": False, "overall_status": "blocked"}, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}
        return {"cmd": list(spec["cmd"]), "rc": 1, "payload": {}, "stdout_tail": "", "stderr_tail": "boom", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["overall_status"] == "blocked"
    assert payload["artifacts_recovered_count"] == 2
    assert payload["blocked_step_count"] == 1
    assert payload["error_step_count"] == 1
    assert payload["missing_before"] == ["ready_artifact", "blocked_artifact", "missing_artifact"]
    assert payload["missing_after"] == ["missing_artifact"]


def test_runtime_artifact_refresh_is_degraded_when_outputs_exist_but_one_is_blocked(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    ready_path = health / "ready_latest.json"
    blocked_path = health / "blocked_latest.json"

    specs = [
        {"name": "ready_artifact", "payload_path": ready_path, "cmd": ["ready"]},
        {"name": "blocked_artifact", "payload_path": blocked_path, "cmd": ["blocked"]},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        name = spec["name"]
        if name == "ready_artifact":
            _write_json(Path(spec["payload_path"]), {"timestamp_utc": "2026-04-22T16:00:00Z", "ok": True, "overall_status": "ready"})
            return {"cmd": list(spec["cmd"]), "rc": 0, "payload": {"ok": True, "overall_status": "ready"}, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}
        _write_json(Path(spec["payload_path"]), {"timestamp_utc": "2026-04-22T16:00:00Z", "ok": False, "overall_status": "blocked"})
        return {"cmd": list(spec["cmd"]), "rc": 2, "payload": {"ok": False, "overall_status": "blocked"}, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["ok"] is True
    assert payload["overall_status"] == "degraded"
    assert payload["missing_after"] == []
    assert payload["error_step_count"] == 0


def test_runtime_artifact_refresh_counts_only_terminal_verifier_for_duplicate_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "governance" / "health" / "fixed_point_latest.json"
    specs = [
        {"name": "fixed_point_precheck", "payload_path": artifact_path, "cmd": ["precheck"]},
        {"name": "fixed_point_verified", "payload_path": artifact_path, "cmd": ["verified"]},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        ready = spec["name"] == "fixed_point_verified"
        payload = {"ok": ready, "overall_status": "ready" if ready else "blocked", "producer": spec["name"]}
        _write_json(Path(spec["payload_path"]), payload)
        return {
            "cmd": list(spec["cmd"]),
            "rc": 0 if ready else 2,
            "payload": payload,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_ms": 1.0,
        }

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["overall_status"] == "ready"
    assert payload["blocked_step_count"] == 0
    assert payload["target_refresh_step_count"] == 2
    assert payload["target_artifact_count"] == 1
    assert payload["artifact_present_count_after"] == 1
    assert payload["superseded_step_count"] == 1
    assert payload["steps"][0]["superseded_by_later_verifier"] is True
    assert payload["steps"][0]["counts_toward_overall"] is False
    assert payload["steps"][1]["counts_toward_overall"] is True


def test_runtime_artifact_refresh_retries_until_artifact_is_current_cycle_fresh(tmp_path: Path) -> None:
    artifact_path = tmp_path / "governance" / "health" / "retry_latest.json"
    _write_json(artifact_path, {"overall_status": "ready", "generation": "old"})
    os.utime(artifact_path, (1_600_000_000, 1_600_000_000))
    specs = [{"name": "retry_artifact", "payload_path": artifact_path, "cmd": ["retry"]}]
    calls = 0

    def runner(spec: dict, project_root: Path) -> dict:
        nonlocal calls
        calls += 1
        payload = {"ok": True, "overall_status": "ready", "generation": f"attempt-{calls}"}
        if calls == 2:
            _write_json(Path(spec["payload_path"]), payload)
        return {"cmd": list(spec["cmd"]), "rc": 0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert calls == 2
    assert payload["overall_status"] == "ready"
    assert payload["all_required_artifacts_fresh"] is True
    assert payload["required_stale_after"] == []
    assert payload["steps"][0]["artifact_refreshed_this_cycle"] is True
    assert payload["steps"][0]["refresh_attempt_count"] == 2


def test_runtime_artifact_refresh_blocks_required_artifact_that_remains_stale(tmp_path: Path) -> None:
    artifact_path = tmp_path / "governance" / "health" / "stale_latest.json"
    old_payload = {"ok": True, "overall_status": "ready", "generation": "old"}
    _write_json(artifact_path, old_payload)
    os.utime(artifact_path, (1_600_000_000, 1_600_000_000))
    specs = [{"name": "stale_artifact", "payload_path": artifact_path, "cmd": ["stale"]}]
    calls = 0

    def runner(spec: dict, project_root: Path) -> dict:
        nonlocal calls
        calls += 1
        return {
            "cmd": list(spec["cmd"]),
            "rc": 0,
            "payload": old_payload,
            "payload_source": "artifact_fallback",
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_ms": 1.0,
        }

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert calls == 2
    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert payload["all_required_artifacts_fresh"] is False
    assert payload["stale_after_refresh"] == ["stale_artifact"]
    assert payload["required_stale_after"] == ["stale_artifact"]
    assert payload["steps"][0]["status"] == "stale"
    assert payload["steps"][0]["failure_envelope_published"] is True
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["stale_source_rejected"] is True


def test_runtime_artifact_refresh_publishes_current_stdout_when_producer_does_not_write(tmp_path: Path) -> None:
    artifact_path = tmp_path / "governance" / "health" / "stdout_latest.json"
    _write_json(artifact_path, {"overall_status": "ready", "generation": "old"})
    os.utime(artifact_path, (1_600_000_000, 1_600_000_000))
    specs = [{"name": "stdout_artifact", "payload_path": artifact_path, "cmd": ["stdout"]}]

    def runner(spec: dict, project_root: Path) -> dict:
        payload = {"ok": True, "overall_status": "ready", "generation": "current"}
        return {
            "cmd": list(spec["cmd"]),
            "rc": 0,
            "payload": payload,
            "payload_source": "stdout",
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_ms": 1.0,
        }

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["overall_status"] == "ready"
    assert payload["steps"][0]["artifact_refreshed_this_cycle"] is True
    assert payload["steps"][0]["refresh_attempt_count"] == 1
    assert payload["steps"][0]["published_from_stdout"] is True
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["generation"] == "current"


def test_runtime_artifact_refresh_blocks_consumers_when_current_epoch_dependency_did_not_publish(
    tmp_path: Path,
) -> None:
    upstream_path = tmp_path / "governance" / "health" / "upstream_latest.json"
    downstream_path = tmp_path / "governance" / "health" / "downstream_latest.json"
    specs = [
        {"name": "upstream", "payload_path": upstream_path, "cmd": ["upstream"]},
        {
            "name": "downstream",
            "payload_path": downstream_path,
            "cmd": ["downstream"],
            "depends_on": ["upstream"],
        },
    ]
    calls: list[str] = []

    def runner(spec: dict, project_root: Path) -> dict:
        calls.append(str(spec["name"]))
        return {
            "cmd": list(spec["cmd"]),
            "rc": 1,
            "payload": {},
            "stdout_tail": "",
            "stderr_tail": "producer failed",
            "duration_ms": 1.0,
        }

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert calls == ["upstream", "upstream"]
    assert payload["overall_status"] == "blocked"
    downstream = json.loads(downstream_path.read_text(encoding="utf-8"))
    assert downstream["dependency_epoch_rejected"] is True
    assert downstream["missing_current_epoch_dependencies"] == ["upstream"]
    assert downstream["evidence_epoch"]["id"] == payload["evidence_epoch_id"]


def test_runtime_artifact_refresh_training_scope_is_dependency_closed(tmp_path: Path) -> None:
    selected = runtime_artifact_refresh._select_scope_specs(
        runtime_artifact_refresh._step_specs(tmp_path),
        "training",
    )
    names = [str(row["name"]) for row in selected]

    assert names == [
        "replay_hash_registry_final",
        "golden_replay_regression_final",
        "runtime_training_snapshot_verified",
        "point_in_time_event_store_verified",
        "feature_store_manifest_verified",
        "training_label_audit_verified",
        "training_lineage_manifest_verified",
        "training_quality_control_verified",
        "bot_needs_intelligence_verified",
        "retrain_schema_compatibility_verified",
        "training_runtime_control_verified",
    ]
    selected_names = set(names)
    assert all(
        set(str(item) for item in row.get("depends_on", [])) <= selected_names
        for row in selected
    )


def test_runtime_artifact_refresh_cell_health_scope_refreshes_every_cell_input(tmp_path: Path) -> None:
    selected = runtime_artifact_refresh._select_scope_specs(
        runtime_artifact_refresh._step_specs(tmp_path),
        "cell-health",
    )
    names = {str(row["name"]) for row in selected}

    assert {
        "cell_sleeve_ticker_universe_pre_intelligence",
        "cell_core_materialization_pre_intelligence",
        "cell_backpressure_super_drainer_pre_intelligence",
        "cell_data_plane_recovery_pre_intelligence",
        "cell_federation_intelligence_pre",
        "cell_whole_system_intelligence",
        "cell_whole_system_governor",
        "cell_sleeve_profitability_dashboard",
        "cell_sleeve_ticker_universe",
        "cell_writer_process_intelligence",
        "cell_backlog_pump_infrabots",
        "cell_training_data_intake",
        "cell_training_labeling",
        "cell_training_probation_isolation",
        "cell_provider_mesh",
        "cell_macro_event_intelligence",
        "cell_watchdog_intelligence",
        "data_collection_observation_rollup_terminal",
        "runtime_gate_dashboard_cell_convergence",
        "cell_infrabot_library_self_awareness_convergence",
        "distributed_cell_architecture_convergence_1",
        "cell_federation_intelligence_convergence",
        "one_numbers_regression_guard_cell_pre",
        "stateful_storage_regression_guard_cell_pre",
        "backlog_organizer_cell_convergence",
        "livefeed_refresh_guard_cell_pre",
        "backlog_pcore_accelerator_cell_pre",
        "backpressure_drainer_fleet_cell_pre",
        "adaptive_regression_guard_cell_convergence",
        "system_architecture_hardening_cell_convergence",
        "health_fast_cell_reconciled",
        "backlog_organizer_cell_verified",
        "incident_closeout_cell_convergence",
        "system_architecture_contract_graph_cell_convergence",
        "system_drift_guard_cell_probe",
        "system_architecture_contract_graph_cell_reconciled",
        "system_architecture_autopilot_cell_convergence",
        "system_architecture_contract_graph_cell_verified",
        "system_drift_guard_cell_convergence",
        "master_infrastructure_supervisor_cell_convergence",
        "system_drift_guard_cell_final",
        "system_architecture_contract_graph_cell_final",
        "cell_platform_brain_v6_convergence",
        "cell_whole_system_intelligence_convergence",
        "cell_whole_system_governor_convergence",
        "distributed_cell_architecture_convergence_2",
        "cell_federation_intelligence_terminal",
        "cell_data_plane_recovery_terminal",
        "cell_whole_system_intelligence_terminal",
        "grade_regression_guard_terminal",
        "section_grade_guard_terminal",
        "low_grade_finalizer_verified",
    } <= names
    assert all(set(str(item) for item in row.get("depends_on", [])) <= names for row in selected)
    steps_by_name = {str(row["name"]): row for row in selected}
    for terminal_name in (
        "cell_platform_brain_v6_convergence",
        "distributed_cell_architecture_convergence_2",
        "grade_regression_guard_terminal",
        "section_grade_guard_terminal",
    ):
        assert "health_fast_cell_reconciled" in steps_by_name[terminal_name]["depends_on"]
        assert "health_fast_terminal" not in steps_by_name[terminal_name]["depends_on"]
    ordered = [str(row["name"]) for row in selected]
    assert ordered.index("cell_sleeve_ticker_universe_pre_intelligence") < ordered.index("cell_whole_system_intelligence")
    assert ordered.index("cell_core_materialization_pre_intelligence") < ordered.index("cell_federation_intelligence_pre")
    assert ordered.index("cell_backpressure_super_drainer_pre_intelligence") < ordered.index("cell_federation_intelligence_pre")
    assert ordered.index("cell_backpressure_super_drainer_pre_intelligence") < ordered.index("cell_data_plane_recovery_pre_intelligence")
    assert ordered.index("cell_data_plane_recovery_pre_intelligence") < ordered.index("cell_federation_intelligence_pre")
    assert ordered.index("cell_federation_intelligence_pre") < ordered.index("cell_whole_system_intelligence")
    assert ordered.index("data_collection_observation_rollup_terminal") < ordered.index("health_fast_terminal")
    assert ordered.index("health_fast_terminal") < ordered.index("runtime_gate_dashboard_cell_convergence")
    assert ordered.index("runtime_gate_dashboard_cell_convergence") < ordered.index("cell_infrabot_library_self_awareness_convergence")
    assert ordered.index("cell_infrabot_library_self_awareness_convergence") < ordered.index("distributed_cell_architecture_convergence_1")
    assert ordered.index("distributed_cell_architecture_convergence_1") < ordered.index("cell_federation_intelligence_convergence")
    assert ordered.index("cell_federation_intelligence_convergence") < ordered.index("adaptive_regression_guard_cell_convergence")
    assert ordered.index("stateful_storage_regression_guard_cell_pre") < ordered.index("adaptive_regression_guard_cell_convergence")
    assert ordered.index("livefeed_refresh_guard_cell_pre") < ordered.index("adaptive_regression_guard_cell_convergence")
    assert ordered.index("backlog_pcore_accelerator_cell_pre") < ordered.index("backpressure_drainer_fleet_cell_pre")
    assert ordered.index("backpressure_drainer_fleet_cell_pre") < ordered.index("adaptive_regression_guard_cell_convergence")
    assert ordered.index("adaptive_regression_guard_cell_convergence") < ordered.index("system_architecture_hardening_cell_convergence")
    assert ordered.index("system_architecture_hardening_cell_convergence") < ordered.index("health_fast_cell_reconciled")
    assert ordered.index("health_fast_cell_reconciled") < ordered.index("backlog_organizer_cell_verified")
    assert ordered.index("backlog_organizer_cell_verified") < ordered.index("system_architecture_contract_graph_cell_convergence")
    assert ordered.index("system_architecture_contract_graph_cell_convergence") < ordered.index("system_drift_guard_cell_probe")
    assert ordered.index("system_drift_guard_cell_probe") < ordered.index("system_architecture_contract_graph_cell_reconciled")
    assert ordered.index("system_architecture_contract_graph_cell_reconciled") < ordered.index("system_architecture_autopilot_cell_convergence")
    assert ordered.index("system_architecture_autopilot_cell_convergence") < ordered.index("system_drift_guard_cell_convergence")
    assert ordered.index("system_drift_guard_cell_convergence") < ordered.index("system_architecture_contract_graph_cell_verified")
    assert ordered.index("system_drift_guard_cell_convergence") < ordered.index("master_infrastructure_supervisor_cell_convergence")
    assert ordered.index("master_infrastructure_supervisor_cell_convergence") < ordered.index("system_drift_guard_cell_final")
    assert ordered.index("system_drift_guard_cell_final") < ordered.index("system_architecture_contract_graph_cell_final")
    assert ordered.index("master_infrastructure_supervisor_cell_convergence") < ordered.index("cell_whole_system_intelligence_convergence")
    assert ordered.index("cell_whole_system_intelligence_convergence") < ordered.index("distributed_cell_architecture_convergence_2")
    assert ordered.index("distributed_cell_architecture_convergence_2") < ordered.index("cell_federation_intelligence_terminal")
    assert ordered.index("cell_federation_intelligence_terminal") < ordered.index("cell_data_plane_recovery_terminal")
    assert ordered.index("cell_data_plane_recovery_terminal") < ordered.index("cell_whole_system_intelligence_terminal")
    assert ordered.index("cell_whole_system_intelligence_terminal") < ordered.index("grade_regression_guard_terminal")
    assert ordered.index("distributed_cell_architecture_convergence_2") < ordered.index("low_grade_finalizer_verified")


def test_runtime_artifact_refresh_uses_collection_operational_projection() -> None:
    status = runtime_artifact_refresh._step_status(
        {
            "rc": 2,
            "payload": {
                "overall_status": "degraded",
                "operational_status": "ready",
                "operational_ok": True,
                "operational_collection": {"status": "ready", "ok": True},
            },
        },
        name="data_collection_observation_rollup_terminal",
    )

    assert status == "ready_operational"


def test_runtime_artifact_refresh_uses_generic_operational_projection() -> None:
    status = runtime_artifact_refresh._step_status(
        {
            "rc": 0,
            "payload": {
                "overall_status": "constrained",
                "ok": False,
                "operational_status": "ready_idle",
                "operational_ok": True,
                "operational_training": {
                    "status": "ready_idle",
                    "ok": True,
                    "controlled_idle_no_candidates": True,
                },
            },
        },
        name="training_runtime_control_verified",
    )

    assert status == "ready_operational"


def test_runtime_artifact_refresh_manages_live_production_readiness_during_green_paper_soak() -> None:
    status = runtime_artifact_refresh._step_status(
        {"rc": 2, "payload": {"overall_status": "blocked", "ok": False}},
        name="production_readiness_control",
        paper_soak_ready=True,
    )

    assert status == "managed_paper_soak"


def test_runtime_artifact_refresh_profitability_scope_includes_every_epoch_input(tmp_path: Path) -> None:
    selected = runtime_artifact_refresh._select_scope_specs(
        runtime_artifact_refresh._step_specs(tmp_path),
        "profitability",
    )
    names = {str(row["name"]) for row in selected}

    assert "execution_queue_stress" in names
    assert "source_verification_verified" in names
    assert "profitability_hardening_control" in names
    assert "profitability_evidence_firewall" in names
    assert "paper_live_data_standard" in names
    assert "control_surface_ownership" in names
    assert "system_role_contract" in names
    firewall = next(row for row in selected if row["name"] == "profitability_evidence_firewall")
    assert set(firewall["depends_on"]) <= names
    freshness = next(row for row in selected if row["name"] == "artifact_freshness_slo_post_master")
    assert "control_surface_ownership" in freshness["depends_on"]
    assert "system_role_contract" in freshness["depends_on"]


def test_runtime_artifact_refresh_requires_secondary_outputs_from_same_producer_to_be_fresh(tmp_path: Path) -> None:
    primary_path = tmp_path / "governance" / "health" / "primary_latest.json"
    secondary_path = tmp_path / "governance" / "health" / "secondary_latest.json"
    _write_json(primary_path, {"overall_status": "ready", "generation": "old"})
    _write_json(secondary_path, {"overall_status": "ready", "generation": "old"})
    os.utime(primary_path, (1_600_000_000, 1_600_000_000))
    os.utime(secondary_path, (1_600_000_000, 1_600_000_000))
    specs = [
        {
            "name": "multi_output",
            "payload_path": primary_path,
            "additional_payload_paths": [secondary_path],
            "cmd": ["multi"],
        }
    ]
    calls = 0

    def runner(spec: dict, project_root: Path) -> dict:
        nonlocal calls
        calls += 1
        payload = {"ok": True, "overall_status": "ready", "generation": f"attempt-{calls}"}
        _write_json(primary_path, payload)
        if calls == 2:
            _write_json(secondary_path, payload)
        return {"cmd": list(spec["cmd"]), "rc": 0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert calls == 2
    assert payload["overall_status"] == "ready"
    assert payload["steps"][0]["artifact_refreshed_this_cycle"] is True
    assert all(payload["steps"][0]["artifact_path_freshness"].values())


def test_runtime_artifact_refresh_treats_managed_production_locks_as_ready(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    specs = [
        {"name": "live_money_readiness_contract", "payload_path": health / "live_money_readiness_contract_latest.json", "cmd": ["live-money"]},
        {"name": "promotion_packet_builder", "payload_path": champion / "promotion_packet_latest.json", "cmd": ["packet"]},
        {"name": "retrain_schema_compatibility", "payload_path": health / "retrain_schema_compatibility_latest.json", "cmd": ["schema"]},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        path = Path(spec["payload_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if spec["name"] == "live_money_readiness_contract":
            payload = {
                "ok": False,
                "overall_status": "blocked",
                "live_money_locked": True,
                "blocking_reasons": ["target_window_not_complete"],
                "grade_summary": {"below_floor_sections": [], "not_ready_sections": []},
            }
            rc = 2
        elif spec["name"] == "promotion_packet_builder":
            payload = {
                "ok": False,
                "promotion_scope": {"target_count": 0, "trained_bot_ids": [], "failure_count": 0},
                "committee_packet_seed_ready": True,
                "replayability_contract": {"hash_bundle_complete": True, "exact_replay_ready": True},
                "gate_results": {
                    "training_success_confirmed": True,
                    "feature_store_manifest_strict_ok": True,
                },
            }
            rc = 2
        else:
            payload = {
                "ok": True,
                "overall_status": "degraded",
                "compatibility_seed_ready": True,
                "failed_checks": [],
                "drifted_fields": [],
            }
            rc = 0
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {"cmd": list(spec["cmd"]), "rc": rc, "payload": payload, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["overall_status"] == "ready"
    assert [row["status"] for row in payload["steps"]] == ["ready_locked", "ready_seeded", "ready_seeded"]


def test_runtime_artifact_refresh_treats_protective_profitability_control_as_ready(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    specs = [
        {"name": "paper_profitability_control", "payload_path": health / "paper_profitability_control_latest.json", "cmd": ["paper-profit"]},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        payload = {
            "ok": True,
            "overall_status": "protective_tightening",
            "controlled_profitability_grade": "A+",
            "profitability_display_grade": "A+ controlled / D raw",
            "raw_profitability_grade": "D",
        }
        _write_json(Path(spec["payload_path"]), payload)
        return {"cmd": list(spec["cmd"]), "rc": 0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["overall_status"] == "ready"
    assert payload["degraded_step_count"] == 0
    assert payload["steps"][0]["status"] == "ready_protective"
    assert payload["steps"][0]["payload_summary"]["raw_profitability_grade"] == "D"


def test_runtime_artifact_refresh_tracks_paper_soak_proof_debt_as_managed(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {"ok": True, "overall_status": "ready", "safe_to_leave_unattended": True},
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {"ok": True, "overall_status": "ready"},
    )
    specs = [
        {"name": "training_quality_control", "payload_path": health / "training_quality_control_latest.json", "cmd": ["training"]},
        {"name": "paper_execution_truth", "payload_path": health / "paper_execution_truth_layer_latest.json", "cmd": ["truth"]},
        {"name": "promotion_packet_builder", "payload_path": tmp_path / "governance" / "champion_challenger" / "promotion_packet_latest.json", "cmd": ["packet"]},
        {"name": "canary_rollout_guard", "payload_path": health / "canary_rollout_latest.json", "cmd": ["canary"], "optional": True},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        if spec["name"] == "canary_rollout_guard":
            return {"cmd": list(spec["cmd"]), "rc": 124, "payload": {}, "stdout_tail": "", "stderr_tail": "timeout", "duration_ms": 1.0}
        if spec["name"] == "promotion_packet_builder":
            payload = {"ok": False, "committee_packet_seed_ready": True, "signing_material_ready": True}
            _write_json(Path(spec["payload_path"]), payload)
            return {"cmd": list(spec["cmd"]), "rc": 2, "payload": payload, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}
        payload = {"ok": False, "overall_status": "blocked", "failed_checks": ["future_live_money_proof"]}
        _write_json(Path(spec["payload_path"]), payload)
        return {"cmd": list(spec["cmd"]), "rc": 2, "payload": payload, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["overall_status"] == "ready"
    assert payload["managed_paper_soak_step_count"] == 4
    assert [row["status"] for row in payload["steps"]] == [
        "managed_paper_soak",
        "managed_paper_soak",
        "managed_paper_soak",
        "managed_paper_soak",
    ]


def test_runtime_artifact_refresh_manages_verified_suffix_for_paper_evidence_debt(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {"ok": True, "overall_status": "ready", "safe_to_leave_unattended": True},
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    artifact = health / "training_quality_control_latest.json"
    specs = [{"name": "training_quality_control_verified", "payload_path": artifact, "cmd": ["training"]}]

    def runner(spec: dict, project_root: Path) -> dict:
        payload = {"ok": False, "overall_status": "needs_attention"}
        _write_json(Path(spec["payload_path"]), payload)
        return {"cmd": list(spec["cmd"]), "rc": 0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["overall_status"] == "ready"
    assert payload["steps"][0]["status"] == "managed_paper_soak"


def test_runtime_artifact_refresh_manages_failed_suffix_step_without_payload() -> None:
    status = runtime_artifact_refresh._step_status(
        {"rc": 124, "payload": {}},
        name="training_quality_control_verified",
        paper_soak_ready=True,
    )

    assert status == "managed_paper_soak"


def test_runtime_artifact_refresh_uses_required_provider_mesh_projection() -> None:
    status = runtime_artifact_refresh._step_status(
        {
            "rc": 0,
            "payload": {
                "ok": False,
                "overall_status": "degraded",
                "required_failures": [],
                "summary": {
                    "required_collectors": 4,
                    "required_contract_ok": 4,
                    "required_snapshot_ready": 4,
                },
            },
        },
        name="cell_provider_mesh",
    )

    assert status == "ready_operational"


def test_runtime_artifact_refresh_treats_ok_true_needs_work_as_advisory_ready() -> None:
    status = runtime_artifact_refresh._step_status(
        {"rc": 0, "payload": {"ok": True, "overall_status": "needs_work"}},
        name="cell_platform_brain_v6_convergence",
    )

    assert status == "ready_advisory"


def test_runtime_artifact_refresh_manages_intentional_live_lock_halt_during_paper_soak(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {"ok": True, "overall_status": "ready", "safe_to_leave_unattended": True},
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    specs = [
        {"name": "halt_trigger_control_plane_terminal", "payload_path": health / "halt_trigger_control_plane_latest.json", "cmd": ["halt"]},
        {"name": "coordination_state_control_terminal", "payload_path": health / "coordination_state_latest.json", "cmd": ["coordination"]},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        if spec["name"].startswith("halt_trigger_control_plane"):
            payload = {
                "overall_status": "blocked",
                "effective_state": "live_read_only",
                "execution_policy": {
                    "paper_trade_lock_active": True,
                    "effective_live_order_execution_allowed": False,
                },
                "manual_flags": {
                    "operator_stop": {"active": False},
                    "global_halt": {"active": False},
                },
                "issues": [
                    {"name": "paper_trade_lock_active"},
                    {"name": "runtime_clearance_not_thaw_safe"},
                    {
                        "name": "advisory_artifact_stale:incident_auto_halt",
                        "blocks_live_execution": False,
                        "blocks_halt_clear": False,
                        "blocks_heavy_viewer": False,
                    },
                ],
            }
        else:
            payload = {"overall_status": "blocked", "ok": False}
        _write_json(Path(spec["payload_path"]), payload)
        return {"cmd": list(spec["cmd"]), "rc": 2, "payload": payload, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["overall_status"] == "ready"
    assert payload["managed_paper_soak_step_count"] == 2
    assert [row["status"] for row in payload["steps"]] == ["managed_paper_soak", "managed_paper_soak"]


def test_runtime_artifact_refresh_rechecks_soak_after_refreshing_core_artifacts(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    specs = [
        {"name": "canary_rollout_guard", "payload_path": health / "canary_rollout_latest.json", "cmd": ["canary"], "optional": True},
        {"name": "training_quality_control", "payload_path": health / "training_quality_control_latest.json", "cmd": ["training"]},
        {"name": "unattended_soak_readiness", "payload_path": health / "unattended_soak_readiness_latest.json", "cmd": ["soak"]},
        {"name": "runtime_paper_regression_guard", "payload_path": health / "runtime_paper_regression_guard_latest.json", "cmd": ["paper"]},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        if spec["name"] == "canary_rollout_guard":
            return {"cmd": list(spec["cmd"]), "rc": 124, "payload": {}, "stdout_tail": "", "stderr_tail": "timeout", "duration_ms": 1.0}
        if spec["name"] == "training_quality_control":
            payload = {"ok": False, "overall_status": "blocked", "failed_checks": ["future_live_money_proof"]}
        elif spec["name"] == "unattended_soak_readiness":
            payload = {"ok": True, "overall_status": "ready", "safe_to_leave_unattended": True}
        else:
            payload = {"ok": True, "overall_status": "ready"}
        _write_json(Path(spec["payload_path"]), payload)
        return {"cmd": list(spec["cmd"]), "rc": 0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["paper_soak_ready_before_refresh"] is False
    assert payload["paper_soak_ready_after_refresh"] is True
    assert payload["overall_status"] == "ready"
    assert payload["required_missing_after"] == []
    assert payload["managed_paper_soak_step_count"] == 2
    assert {row["name"]: row["status"] for row in payload["steps"]}["canary_rollout_guard"] == "managed_paper_soak"
    assert {row["name"]: row["status"] for row in payload["steps"]}["training_quality_control"] == "managed_paper_soak"


def test_runtime_artifact_refresh_manages_stateful_sql_soft_quota_during_green_soak(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {"ok": True, "overall_status": "ready", "safe_to_leave_unattended": True},
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure": {
                "raw_live": {
                    "core_pending_lines": 28,
                    "total_pending_lines": 28,
                    "oldest_pending_age_seconds": 0.0,
                }
            },
        },
    )
    _write_json(
        health / "storage_retention_unison_latest.json",
        {
            "overall_status": "ready",
            "continuous_run_contract": {"ready": True, "storage_controls": {"quota_ready": True}},
            "storage_growth_forecast": {"status": "stable_or_improving", "days_until_pressure_free": 45},
            "integration_contract": {"stateful_sql_compaction_only": True},
        },
    )
    _write_json(
        health / "storage_tier_policy_latest.json",
        {
            "overall_status": "advisory",
            "manifest_backed_offload_contract": {
                "stateful_sql_policy": "checkpoint and compact stateful SQL; never source-delete from this policy"
            },
        },
    )
    specs = [
        {"name": "storage_quota_guard", "payload_path": health / "storage_quota_guard_latest.json", "cmd": ["quota"]},
    ]

    def runner(spec: dict, project_root: Path) -> dict:
        payload = {
            "ok": False,
            "overall_status": "degraded",
            "quota_summary": {
                "hard_breaches": 0,
                "soft_breaches": 1,
                "blocked_families": [],
                "degraded_families": ["sql_link_shards"],
                "worst_over_hard_gb": 0.0,
                "worst_hard_ratio": 0.855,
            },
            "lanes": [
                {
                    "family": "sql_link_shards",
                    "status": "degraded",
                    "over_hard_gb": 0.0,
                    "hard_ratio": 0.855,
                }
            ],
        }
        _write_json(Path(spec["payload_path"]), payload)
        return {"cmd": list(spec["cmd"]), "rc": 2, "payload": payload, "stdout_tail": "", "stderr_tail": "", "duration_ms": 1.0}

    payload = runtime_artifact_refresh.build_payload(tmp_path, specs=specs, runner=runner)

    assert payload["overall_status"] == "ready"
    assert payload["degraded_step_count"] == 0
    assert payload["managed_paper_soak_step_count"] == 1
    assert payload["steps"][0]["status"] == "managed_paper_soak"


def test_runtime_artifact_refresh_step_specs_include_training_storage_and_hardening_contracts(tmp_path: Path) -> None:
    specs = runtime_artifact_refresh._step_specs(tmp_path)
    names = [row["name"] for row in specs]

    assert "training_lineage_manifest" in names
    assert "training_quality_control" in names
    assert "portfolio_capacity_curve_report" in names
    assert "cross_host_parity_report" in names
    assert "cost_telemetry" in names
    assert "broker_readiness" in names
    assert "session_ready" in names
    assert "storage_failback_sync" in names
    assert "promotion_autopilot_packet" in names
    assert "source_verification" in names
    assert "capability_materialization" in names
    assert names.index("source_verification_verified") < names.index("capability_materialization")
    assert names.index("capability_materialization") < names.index("collector_capability_control")
    assert "paper_performance" in names
    assert "paper_live_data_standard" in names
    assert "paper_profitability_control" in names
    assert names.index("paper_performance") < names.index("paper_profitability_control")
    assert "paper_replay_drill" in names
    assert "paper_execution_truth" in names
    assert "retrain_schema_compatibility" in names
    assert "promotion_packet_builder" in names
    assert "promotion_quality_gate" in names
    assert "canary_rollout_guard" in names
    assert "ingestion_storage_control" in names
    assert "storage_resilience_control" in names
    assert "storage_retention_unison" in names
    assert "security_evidence_autofix" in names
    assert "security_audit" in names
    assert "incident_closeout_autopilot" in names
    assert "live_canary_control" in names
    assert "live_readiness_smoke" in names
    assert "live_money_readiness_contract" in names
    assert "runtime_throttle_control" in names
    assert "runtime_paper_regression_guard" in names
    assert "ingestion_backpressure_final" in names
    assert "ingestion_storage_control_final" in names
    assert "ingestion_storage_governor_final" in names
    assert "ingestion_storage_control_post_governor" in names
    assert "ingestion_storage_governor_verify" in names
    assert "ingestion_storage_control_verified" in names
    assert "regime_control_plane" in names
    assert "market_cycle_extraction_engine" in names
    assert "chrome_headless_guard" in names
    assert "multiple_testing_guard" in names
    assert "decay_monitor" in names
    assert "execution_queue_stress" in names
    assert "profitability_independent_validator" in names
    assert "profitability_holdout_vault" in names
    assert "profitability_benchmark_capture" in names
    assert "profitability_benchmark_hurdle" in names
    assert "health_gates" in names
    assert "health_fast" in names
    assert "service_control_plane" in names
    assert "runtime_throttle_control_verified" in names
    assert "paper_400_ramp_verified" in names
    assert "runtime_paper_regression_guard_verified" in names
    assert "halt_trigger_control_plane_verified" in names
    assert "coordination_state_control_verified" in names
    assert "health_fast_verified" in names
    assert "unattended_soak_readiness_verified" in names
    assert "replay_hash_registry_final" in names
    assert "golden_replay_regression_final" in names
    assert "paper_execution_truth_verified" in names
    assert "stateful_storage_regression_guard_verified" in names
    assert "one_numbers_regression_guard_verified" in names
    assert "grade_regression_guard_verified" in names
    assert "section_grade_guard_verified" in names
    assert "low_grade_finalizer_verified" in names
    assert "system_drift_registry_verified" in names
    assert "codex_project_guard_verified" in names
    assert "coinbase_api_health_verified" in names
    assert "infrastructure_autofix_verified" in names
    assert "master_infrastructure_supervisor_verified" in names
    assert "process_watchdog_verified" in names
    assert "livefeed_refresh_guard_verified" in names
    assert "runtime_throttle_control_final" in names
    assert "paper_400_ramp_final" in names
    assert "runtime_paper_regression_guard_final" in names
    assert "halt_trigger_control_plane_final" in names
    assert "coordination_state_control_final" in names
    assert "health_fast_final" in names
    assert "adaptive_regression_guard_final" in names
    assert "system_drift_guard_pre_architecture" in names
    assert "cell_whole_system_intelligence" in names
    assert "cell_sleeve_ticker_universe" in names
    assert "cell_training_labeling" in names
    assert "cell_provider_mesh" in names
    assert "cell_watchdog_intelligence" in names
    assert "schwab_indicator_intelligence_verified" in names
    assert "system_expansion_execution_verified" in names
    assert "distributed_cell_architecture_verified" in names
    assert "system_architecture_hardening_verified" in names
    assert "system_self_model_pre_architecture" in names
    assert "system_architecture_contract_graph_final" in names
    assert "system_architecture_autopilot_final" in names
    assert "system_drift_guard_verified" in names
    assert "system_drift_autopilot_verified" in names
    assert "master_infrastructure_supervisor_final" in names
    assert "system_self_model_verified" in names
    assert "system_architecture_contract_graph_verified" in names
    assert "system_architecture_autopilot_verified" in names
    assert "broker_readiness_terminal" in names
    assert "auth_lease_manager_terminal" in names
    assert "schwab_auth_supervisor_terminal" in names
    assert "backlog_pcore_accelerator_terminal" in names
    assert "backpressure_drainer_fleet_terminal" in names
    assert "ingestion_storage_control_terminal" in names
    assert "runtime_throttle_control_terminal" in names
    assert "paper_400_ramp_terminal" in names
    assert "runtime_paper_regression_guard_terminal" in names
    assert "halt_trigger_control_plane_terminal" in names
    assert "coordination_state_control_terminal" in names
    assert "health_fast_terminal" in names
    assert "unattended_soak_readiness_terminal" in names
    assert "grade_regression_guard_terminal" in names
    assert "section_grade_guard_terminal" in names
    assert "livefeed_refresh_guard_terminal" in names
    assert "adaptive_regression_guard_terminal" in names
    assert "system_self_model_convergence" in names
    assert "system_architecture_contract_graph_convergence" in names
    assert "system_architecture_autopilot_convergence" in names
    assert "system_drift_guard_terminal" in names
    assert "system_drift_autopilot_terminal" in names
    assert "system_drift_guard_post_autopilot_terminal" in names
    assert "one_numbers_regression_guard_terminal" in names
    assert "master_infrastructure_supervisor_terminal" in names
    assert "infrastructure_autofix_terminal" in names
    assert "system_self_model_final" in names
    assert "system_architecture_contract_graph_terminal" in names
    assert "system_architecture_autopilot_terminal" in names
    assert "system_drift_guard_settled" in names
    assert "system_drift_autopilot_settled" in names
    assert "system_drift_guard_post_settled" in names
    assert "master_infrastructure_supervisor_settled" in names
    assert "infrastructure_autofix_settled" in names
    assert "system_self_model_settled" in names
    assert "system_architecture_contract_graph_settled" in names
    assert "system_architecture_autopilot_settled" in names
    assert "system_drift_guard_final" in names
    assert "master_infrastructure_supervisor_final_settled" in names
    assert "bot_needs_intelligence" in names
    assert "training_runtime_control" in names
    assert "storage_disaster_recovery" in names
    assert "chaos_drill_coordinator" in names
    assert "live_order_ledger_control" in names
    assert "content_addressed_artifact_store" in names
    assert "storage_disaster_recovery_verified" in names
    assert "security_evidence_autofix_verified" in names
    assert "security_audit_verified" in names
    assert "remote_alert_control_verified" in names
    assert "blackstart_recovery_verified" in names
    assert "telemetry_redaction_canary_verified" in names

    assert names.index("storage_quota_guard") < names.index("state_snapshot_restore_drill")
    assert names.index("state_snapshot_restore_drill") < names.index("storage_resilience_control")
    assert names.index("storage_quota_guard") < names.index("ingestion_storage_control")
    assert names.index("ingestion_storage_control") < names.index("storage_pressure_clearance")
    assert names.index("storage_pressure_clearance") < names.index("unattended_soak_readiness")
    assert names.index("storage_resilience_control") < names.index("storage_retention_unison")
    assert names.index("storage_retention_unison") < names.index("storage_resilience_control_terminal")
    assert names.index("storage_resilience_control_terminal") < names.index("blackstart_recovery")
    assert names.index("storage_retention_unison") < names.index("unattended_soak_readiness")
    assert names.index("content_addressed_artifact_store") < names.index("storage_disaster_recovery_verified")
    assert names.index("security_evidence_autofix_verified") < names.index("security_audit_verified")
    assert names.index("storage_disaster_recovery_verified") < names.index("blackstart_recovery_verified")
    assert names.index("security_audit_verified") < names.index("production_readiness_control")
    assert names.index("remote_alert_control_verified") < names.index("production_readiness_control")
    assert names.index("blackstart_recovery_verified") < names.index("production_readiness_control")
    assert names.index("telemetry_redaction_canary_verified") < names.index("production_readiness_control")
    assert names.index("storage_disaster_recovery_verified") < names.index("production_readiness_control")
    assert names.index("live_readiness_smoke") < names.index("blackstart_recovery")
    assert names.index("blackstart_recovery") < names.index("incident_timeline")
    assert names.index("incident_timeline") < names.index("incident_review_packet")
    assert names.index("incident_review_packet") < names.index("incident_closeout_autopilot")
    assert names.index("runtime_throttle_control") < names.index("paper_400_ramp")
    assert names.index("paper_400_ramp") < names.index("runtime_paper_regression_guard")
    assert names.index("runtime_paper_regression_guard") < names.index("health_fast")
    assert names.index("live_runtime_separation_control") < names.index("live_canary_control")
    assert names.index("live_canary_control") < names.index("live_readiness_smoke")
    assert names.index("blackstart_recovery") < names.index("live_money_readiness_contract")
    assert names.index("live_money_readiness_contract") < names.index("incident_timeline")
    assert names.index("incident_closeout_autopilot") < names.index("unattended_soak_readiness")
    assert names.index("incident_closeout_autopilot") < names.index("ingestion_backpressure_final")
    assert names.index("ingestion_backpressure_final") < names.index("ingestion_storage_control_final")
    assert names.index("ingestion_storage_control_final") < names.index("ingestion_storage_governor_final")
    assert names.index("ingestion_storage_governor_final") < names.index("ingestion_storage_control_post_governor")
    assert names.index("ingestion_storage_control_post_governor") < names.index("ingestion_storage_governor_verify")
    assert names.index("ingestion_storage_governor_verify") < names.index("ingestion_storage_control_verified")
    assert names.index("ingestion_storage_control_verified") < names.index("health_fast")
    assert names.index("health_fast") < names.index("unattended_soak_readiness")
    assert names.index("unattended_soak_readiness") < names.index("health_gates")
    assert names.index("health_gates") < names.index("service_control_plane")
    assert names.index("health_fast") < names.index("service_control_plane")
    assert names.index("service_control_plane") < names.index("runtime_throttle_control_verified")
    assert names.index("runtime_throttle_control_verified") < names.index("paper_400_ramp_verified")
    assert names.index("paper_400_ramp_verified") < names.index("runtime_paper_regression_guard_verified")
    assert names.index("runtime_paper_regression_guard_verified") < names.index("halt_trigger_control_plane_verified")
    assert names.index("halt_trigger_control_plane_verified") < names.index("coordination_state_control_verified")
    assert names.index("coordination_state_control_verified") < names.index("health_fast_verified")
    assert names.index("health_fast_verified") < names.index("unattended_soak_readiness_verified")
    assert names.index("unattended_soak_readiness_verified") < names.index("replay_hash_registry_final")
    assert names.index("replay_hash_registry_final") < names.index("golden_replay_regression_final")
    assert names.index("golden_replay_regression_final") < names.index("paper_execution_truth_verified")
    assert names.index("paper_execution_truth_verified") < names.index("stateful_storage_regression_guard_verified")
    assert names.index("stateful_storage_regression_guard_verified") < names.index("one_numbers_regression_guard_verified")
    assert names.index("one_numbers_regression_guard_verified") < names.index("grade_regression_guard_verified")
    assert names.index("grade_regression_guard_verified") < names.index("section_grade_guard_verified")
    assert names.index("section_grade_guard_verified") < names.index("system_drift_registry_verified")
    assert names.index("system_drift_registry_verified") < names.index("infrastructure_autofix_verified")
    assert names.index("infrastructure_autofix_verified") < names.index("master_infrastructure_supervisor_verified")
    assert names.index("master_infrastructure_supervisor_verified") < names.index("livefeed_refresh_guard_verified")
    assert names.index("livefeed_refresh_guard_verified") < names.index("adaptive_regression_guard_final")
    assert names.index("livefeed_refresh_guard_verified") < names.index("runtime_throttle_control_final")
    assert names.index("runtime_throttle_control_final") < names.index("paper_400_ramp_final")
    assert names.index("paper_400_ramp_final") < names.index("runtime_paper_regression_guard_final")
    assert names.index("runtime_paper_regression_guard_final") < names.index("halt_trigger_control_plane_final")
    assert names.index("halt_trigger_control_plane_final") < names.index("coordination_state_control_final")
    assert names.index("coordination_state_control_final") < names.index("health_fast_final")
    assert names.index("health_fast_final") < names.index("adaptive_regression_guard_final")
    assert names.index("adaptive_regression_guard_final") < names.index("system_drift_guard_pre_architecture")
    assert names.index("system_drift_guard_pre_architecture") < names.index("schwab_indicator_intelligence_verified")
    assert names.index("schwab_indicator_intelligence_verified") < names.index("system_expansion_execution_verified")
    assert names.index("system_expansion_execution_verified") < names.index("distributed_cell_architecture_verified")
    assert names.index("distributed_cell_architecture_verified") < names.index("system_architecture_hardening_verified")
    assert names.index("system_architecture_hardening_verified") < names.index("system_self_model_pre_architecture")
    assert names.index("system_self_model_pre_architecture") < names.index("system_architecture_contract_graph_final")
    assert names.index("system_architecture_contract_graph_final") < names.index("system_architecture_autopilot_final")
    assert names.index("system_architecture_autopilot_final") < names.index("system_drift_guard_verified")
    assert names.index("system_drift_guard_verified") < names.index("system_drift_autopilot_verified")
    assert names.index("system_drift_autopilot_verified") < names.index("master_infrastructure_supervisor_final")
    assert names.index("master_infrastructure_supervisor_final") < names.index("system_self_model_verified")
    assert names.index("system_self_model_verified") < names.index("system_architecture_contract_graph_verified")
    assert names.index("system_architecture_contract_graph_verified") < names.index("system_architecture_autopilot_verified")
    assert names.index("system_architecture_autopilot_verified") < names.index("broker_readiness_terminal")
    assert names.index("broker_readiness_terminal") < names.index("auth_lease_manager_terminal")
    assert names.index("auth_lease_manager_terminal") < names.index("schwab_auth_supervisor_terminal")
    assert names.index("schwab_auth_supervisor_terminal") < names.index("backlog_pcore_accelerator_terminal")
    assert names.index("backlog_pcore_accelerator_terminal") < names.index("backpressure_drainer_fleet_terminal")
    assert names.index("backpressure_drainer_fleet_terminal") < names.index("ingestion_storage_control_terminal")
    assert names.index("ingestion_storage_control_terminal") < names.index("runtime_throttle_control_terminal")
    assert names.index("runtime_throttle_control_terminal") < names.index("paper_400_ramp_terminal")
    assert names.index("paper_400_ramp_terminal") < names.index("runtime_paper_regression_guard_terminal")
    assert names.index("runtime_paper_regression_guard_terminal") < names.index("halt_trigger_control_plane_terminal")
    assert names.index("halt_trigger_control_plane_terminal") < names.index("coordination_state_control_terminal")
    assert names.index("coordination_state_control_terminal") < names.index("health_fast_terminal")
    assert names.index("health_fast_terminal") < names.index("unattended_soak_readiness_terminal")
    assert names.index("unattended_soak_readiness_terminal") < names.index("one_numbers_regression_guard_terminal")
    assert names.index("one_numbers_regression_guard_terminal") < names.index("grade_regression_guard_terminal")
    assert names.index("grade_regression_guard_terminal") < names.index("section_grade_guard_terminal")
    assert names.index("section_grade_guard_terminal") < names.index("low_grade_finalizer_verified")
    assert names.index("low_grade_finalizer_verified") < names.index("livefeed_refresh_guard_terminal")
    assert names.index("livefeed_refresh_guard_terminal") < names.index("adaptive_regression_guard_terminal")
    assert names.index("adaptive_regression_guard_terminal") < names.index("system_self_model_convergence")
    assert names.index("system_self_model_convergence") < names.index("system_architecture_contract_graph_convergence")
    assert names.index("system_architecture_contract_graph_convergence") < names.index("system_architecture_autopilot_convergence")
    assert names.index("system_architecture_autopilot_convergence") < names.index("system_drift_guard_terminal")
    assert names.index("system_drift_guard_terminal") < names.index("system_drift_autopilot_terminal")
    assert names.index("system_drift_autopilot_terminal") < names.index("system_drift_guard_post_autopilot_terminal")
    assert names.index("system_drift_guard_post_autopilot_terminal") < names.index("master_infrastructure_supervisor_terminal")
    assert names.index("master_infrastructure_supervisor_terminal") < names.index("infrastructure_autofix_terminal")
    assert names.index("infrastructure_autofix_terminal") < names.index("system_self_model_final")
    assert names.index("system_self_model_final") < names.index("system_architecture_contract_graph_terminal")
    assert names.index("system_architecture_contract_graph_terminal") < names.index("system_architecture_autopilot_terminal")
    assert names.index("system_architecture_autopilot_terminal") < names.index("system_drift_guard_settled")
    assert names.index("system_drift_guard_settled") < names.index("system_drift_autopilot_settled")
    assert names.index("system_drift_autopilot_settled") < names.index("system_drift_guard_post_settled")
    assert names.index("system_drift_guard_post_settled") < names.index("master_infrastructure_supervisor_settled")
    assert names.index("master_infrastructure_supervisor_settled") < names.index("infrastructure_autofix_settled")
    assert names.index("infrastructure_autofix_settled") < names.index("system_self_model_settled")
    assert names.index("system_self_model_settled") < names.index("system_architecture_contract_graph_settled")
    assert names.index("system_architecture_contract_graph_settled") < names.index("system_architecture_autopilot_settled")
    assert names.index("system_architecture_autopilot_settled") < names.index("system_drift_guard_final")
    assert names.index("system_drift_guard_final") < names.index("master_infrastructure_supervisor_final_settled")
    assert names.index("master_infrastructure_supervisor_final_settled") < names.index("broker_readiness_post_settlement")
    assert names.index("broker_readiness_post_settlement") < names.index("auth_lease_manager_post_settlement")
    assert names.index("auth_lease_manager_post_settlement") < names.index("schwab_auth_supervisor_post_settlement")
    assert names.index("schwab_auth_supervisor_post_settlement") < names.index("backpressure_drainer_fleet_post_settlement")
    assert names.index("backpressure_drainer_fleet_post_settlement") < names.index("ingestion_storage_control_post_settlement")
    assert names.index("ingestion_storage_control_post_settlement") < names.index("runtime_throttle_control_post_settlement")
    assert names.index("runtime_throttle_control_post_settlement") < names.index("paper_400_ramp_post_settlement")
    assert names.index("paper_400_ramp_post_settlement") < names.index("runtime_throttle_control_post_settlement_verified")
    assert names.index("runtime_throttle_control_post_settlement_verified") < names.index("paper_400_ramp_post_settlement_verified")
    assert names.index("paper_400_ramp_post_settlement_verified") < names.index("runtime_paper_regression_guard_post_settlement")
    assert names.index("runtime_paper_regression_guard_post_settlement") < names.index("halt_trigger_control_plane_post_settlement")
    assert names.index("halt_trigger_control_plane_post_settlement") < names.index("coordination_state_control_post_settlement")
    assert names.index("coordination_state_control_post_settlement") < names.index("health_fast_post_settlement")
    assert names.index("health_fast_post_settlement") < names.index("live_runtime_separation_post_settlement")
    assert names.index("live_runtime_separation_post_settlement") < names.index("incident_closeout_autopilot_post_settlement")
    assert names.index("incident_closeout_autopilot_post_settlement") < names.index("sleeve_isolation_guard_post_settlement")
    assert names.index("sleeve_isolation_guard_post_settlement") < names.index("section_grade_guard_post_settlement")
    assert names.index("section_grade_guard_post_settlement") < names.index("unattended_soak_readiness_post_settlement")
    assert names.index("unattended_soak_readiness_post_settlement") < names.index("livefeed_refresh_guard_post_settlement")
    assert names.index("livefeed_refresh_guard_post_settlement") < names.index("adaptive_regression_guard_post_settlement")
    assert names.index("adaptive_regression_guard_post_settlement") < names.index("system_drift_guard_post_evidence_probe")
    assert names.index("system_drift_guard_post_evidence_probe") < names.index("system_architecture_contract_graph_post_evidence_probe")
    assert names.index("system_architecture_contract_graph_post_evidence_probe") < names.index("system_drift_guard_post_evidence_reconciled")
    assert names.index("system_drift_guard_post_evidence_reconciled") < names.index("system_architecture_contract_graph_post_evidence_verified")
    assert names.index("system_architecture_contract_graph_post_evidence_verified") < names.index("system_architecture_autopilot_post_evidence_verified")
    assert names.index("system_architecture_autopilot_post_evidence_verified") < names.index("system_drift_guard_post_architecture_verified")
    assert names.index("system_drift_guard_post_architecture_verified") < names.index("system_drift_autopilot_post_evidence_verified")
    assert names.index("source_verification") < names.index("source_verification_autorefresh")
    assert names.index("source_verification_autorefresh") < names.index("source_verification_verified")
    assert names.index("multiple_testing_guard") < names.index("profitability_evidence_firewall")
    assert names.index("decay_monitor") < names.index("profitability_evidence_firewall")
    assert names.index("profitability_independent_validator") < names.index("profitability_benchmark_hurdle")
    assert names.index("profitability_benchmark_capture") < names.index("profitability_benchmark_hurdle")
    assert names.index("profitability_benchmark_hurdle") < names.index("profitability_evidence_firewall")
    assert names.index("profitability_evidence_firewall") < names.index("production_excellence_control")
    assert names.index("production_excellence_control") < names.index("continuous_soak_integrity_control")
    assert names.index("continuous_soak_integrity_control") < names.index("live_transition_integrity_control")
    assert names.index("live_transition_integrity_control") < names.index("live_money_readiness_contract_verified")
    assert names.index("live_money_readiness_contract_verified") < names.index("runtime_gate_dashboard_pre_master")
    assert names.index("system_drift_autopilot_post_evidence_verified") < names.index("runtime_gate_dashboard_pre_master")
    assert names.index("runtime_gate_dashboard_pre_master") < names.index("master_infrastructure_supervisor_post_evidence_probe")
    assert names.index("master_infrastructure_supervisor_post_evidence_probe") < names.index("infrastructure_autofix_post_evidence_verified")
    assert names.index("infrastructure_autofix_post_evidence_verified") < names.index("system_self_model_post_evidence_verified")
    assert names.index("system_self_model_post_evidence_verified") < names.index("master_infrastructure_supervisor_post_evidence_verified")
    assert names.index("master_infrastructure_supervisor_post_evidence_verified") < names.index("runtime_gate_dashboard_pre_operator")
    assert names.index("runtime_gate_dashboard_pre_operator") < names.index("operator_cockpit")
    assert names.index("system_architecture_contract_graph_verified") < names.index("operator_cockpit")

    account_snapshot_spec = next(row for row in specs if row["name"] == "schwab_account_snapshot_refresh")
    assert account_snapshot_spec["cmd"][0].endswith("scripts/ops/opsctl.sh")
    assert account_snapshot_spec["cmd"][1] == "schwab-account-snapshot-refresh"
    assert names.index("position_opportunity_watch") < names.index("sleeve_allocator")
    assert names.index("position_opportunity_watch") < names.index("one_numbers_portfolio_prerequisite")
    assert names.index("one_numbers_portfolio_prerequisite") < names.index("sleeve_allocator")
    assert names.index("sleeve_allocator") < names.index("portfolio_risk_ledger")
    assert names.index("portfolio_risk_ledger") < names.index("portfolio_allocator_service")
    assert names.index("portfolio_allocator_service") < names.index("account_buildout_plan")
    chrome_spec = next(row for row in specs if row["name"] == "chrome_headless_guard")
    assert "--apply" in chrome_spec["cmd"]
    paper_profitability_spec = next(row for row in specs if row["name"] == "paper_profitability_control")
    assert "--apply" in paper_profitability_spec["cmd"]
    assert paper_profitability_spec["additional_payload_paths"]
    paper_performance_spec = next(row for row in specs if row["name"] == "paper_performance")
    assert "--json-only" in paper_performance_spec["cmd"]
    source_refresh_spec = next(row for row in specs if row["name"] == "source_verification_autorefresh")
    assert source_refresh_spec["optional"] is True
    assert source_refresh_spec["cmd"][source_refresh_spec["cmd"].index("--max-commands") + 1] == "1"
    security_refresh_spec = next(row for row in specs if row["name"] == "security_evidence_autofix_verified")
    assert "--force-secret-scan" in security_refresh_spec["cmd"]
    assert security_refresh_spec["additional_payload_paths"]
    readiness_spec = next(row for row in specs if row["name"] == "production_readiness_control")
    assert set(readiness_spec["depends_on"]) == {
        "storage_disaster_recovery_verified",
        "security_audit_verified",
        "remote_alert_control_verified",
        "blackstart_recovery_verified",
        "telemetry_redaction_canary_verified",
    }
    indicator_spec = next(row for row in specs if row["name"] == "schwab_indicator_intelligence_verified")
    assert "--offline" in indicator_spec["cmd"]
    runtime_throttle_spec = next(row for row in specs if row["name"] == "runtime_throttle_control")
    assert "--apply" in runtime_throttle_spec["cmd"]
    ingestion_governor_spec = next(row for row in specs if row["name"] == "ingestion_storage_governor_final")
    assert "apply" in ingestion_governor_spec["cmd"]
    ingestion_governor_verify_spec = next(row for row in specs if row["name"] == "ingestion_storage_governor_verify")
    assert "apply" in ingestion_governor_verify_spec["cmd"]
