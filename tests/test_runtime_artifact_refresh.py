import json
from pathlib import Path

from scripts.ops import runtime_artifact_refresh


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    assert "paper_profitability_control" in names
    assert "paper_replay_drill" in names
    assert "paper_execution_truth" in names
    assert "retrain_schema_compatibility" in names
    assert "promotion_packet_builder" in names
    assert "promotion_quality_gate" in names
    assert "canary_rollout_guard" in names
    assert "ingestion_storage_control" in names
    assert "storage_resilience_control" in names
    assert "security_evidence_autofix" in names
    assert "security_audit" in names
    assert "incident_closeout_autopilot" in names
    assert "live_canary_control" in names
    assert "live_readiness_smoke" in names
    assert "live_money_readiness_contract" in names
    assert "runtime_throttle_control" in names
    assert "regime_control_plane" in names
    assert "market_cycle_extraction_engine" in names
    assert "chrome_headless_guard" in names
    assert "multiple_testing_guard" in names

    chrome_spec = next(row for row in specs if row["name"] == "chrome_headless_guard")
    assert "--apply" in chrome_spec["cmd"]
