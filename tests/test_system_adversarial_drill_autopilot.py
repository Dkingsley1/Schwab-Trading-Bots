import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import system_adversarial_drill_autopilot as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_ready(path: Path, *, ok: bool = True) -> None:
    _write_json(path, {"timestamp_utc": src.iso_now(), "overall_status": "ready", "ok": ok})


def test_adversarial_drill_detects_cross_layer_weak_points(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"

    _write_json(
        health / "health_fast_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "operational_readiness": {
                "guarded_paper": {
                    "status": "blocked",
                    "blockers": ["runtime_status=degraded", "memory_status=needs_work"],
                }
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "host_saturation_score": 59.75,
            "throttle_profile": "sustain",
            "soft_cap_advisory_reclassification": {
                "measurements": {
                    "storage_writer_cpu_percent": 255.0,
                    "storage_writer_hot": True,
                }
            },
        },
    )
    _write_json(health / "memory_efficiency_control_latest.json", {"timestamp_utc": src.iso_now(), "overall_status": "needs_work"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "pressure_index": 0.452,
            "backpressure_quality_score": 37.85,
        },
    )
    _write_json(
        health / "storage_pressure_clearance_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "metrics": {
                "active_storage_pressure": True,
                "core_pending_lines": 6774,
                "total_pending_lines": 7762,
            },
        },
    )
    _write_json(
        health / "incident_closeout_autopilot_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "open_incident_count": 2,
            "bounded_closeout_path_ready": True,
            "closeout_score": 92,
        },
    )
    _write_json(
        health / "live_canary_control_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "recommended_mode": "preapproved_supervised",
            "preapproved_supervised_ready": True,
            "supervised_canary_ready": False,
            "preclearance_score": 100,
            "blocking_reasons": ["promotion_packet_preclearance_only"],
        },
    )
    _write_json(
        health / "system_drift_guard_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "blocked",
            "metrics": {
                "blocked_surface_count": 5,
                "degraded_surface_count": 9,
                "stale_surface_count": 5,
            },
        },
    )
    _write_json(
        health / "system_architecture_contract_graph_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "blocked",
            "blocked_node_count": 1,
            "degraded_node_count": 6,
        },
    )
    _write_json(
        health / "architecture_upgrade_scoreboard_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "ready_count": 8,
        },
    )
    _write_json(
        health / "master_infrastructure_supervisor_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "blocked",
            "metrics": {"blocked_check_count": 2, "degraded_check_count": 4},
            "platform_posture": {"operating_posture": "repair_first"},
        },
    )
    _write_json(
        champion / "promotion_autopilot_packet_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "promotion_ready": False,
            "repairable_gate_count": 2,
            "packet_completeness_score": 96,
            "readiness_repair_contract": {"critical_repair_gate_count": 0},
        },
    )
    _seed_ready(health / "command_validity_latest.json")
    _seed_ready(health / "golden_replay_regression_latest.json")
    _seed_ready(health / "replay_hash_registry_guard_latest.json")
    _seed_ready(health / "point_in_time_event_store_latest.json")
    _seed_ready(health / "operator_cockpit_latest.json")
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"timestamp_utc": src.iso_now(), "overall_status": "ready", "ok": True})

    payload = src.build_payload(tmp_path, apply=True)
    weak_ids = {row["weak_point_id"] for row in payload["weak_points"]}

    assert payload["overall_status"] == "blocked"
    assert payload["critical_weak_point_count"] == 1
    assert {
        "guarded_paper_pressure_coupling",
        "sql_writer_heat",
        "raw_live_storage_headroom",
        "bounded_incident_closeout",
        "live_canary_validate_only",
        "artifact_drift_mesh",
        "architecture_contract_pressure",
        "self_auditing_infra_bots",
        "promotion_packet_repairable_gates",
    } <= weak_ids
    assert (tmp_path / "governance" / "drills" / "system_adversarial_drill_results_latest.json").exists()


def test_adversarial_drill_accepts_operator_gated_command_surface(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"

    _seed_ready(health / "health_fast_latest.json")
    _seed_ready(health / "runtime_throttle_control_latest.json")
    _seed_ready(health / "memory_efficiency_control_latest.json")
    _seed_ready(health / "ingestion_storage_control_latest.json")
    _seed_ready(health / "storage_pressure_clearance_latest.json")
    _seed_ready(health / "runtime_paper_regression_guard_latest.json")
    _seed_ready(health / "system_drift_guard_latest.json")
    _seed_ready(health / "system_architecture_contract_graph_latest.json")
    _seed_ready(health / "architecture_upgrade_scoreboard_latest.json")
    _seed_ready(health / "incident_closeout_autopilot_latest.json")
    _write_json(
        health / "live_canary_control_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "ready",
            "ok": True,
            "supervised_canary_ready": True,
        },
    )
    _seed_ready(health / "master_infrastructure_supervisor_latest.json")
    _seed_ready(health / "golden_replay_regression_latest.json")
    _seed_ready(health / "replay_hash_registry_guard_latest.json")
    _seed_ready(health / "point_in_time_event_store_latest.json")
    _seed_ready(health / "operator_cockpit_latest.json")
    _write_json(
        champion / "promotion_autopilot_packet_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "ready",
            "ok": True,
            "promotion_ready": True,
            "repairable_gate_count": 0,
            "readiness_repair_contract": {"critical_repair_gate_count": 0},
        },
    )
    _write_json(
        health / "command_validity_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "degraded",
            "ok": True,
            "metrics": {
                "operator_gated_entry_count": 56,
                "blocked_entry_count": 0,
                "degraded_entry_count": 0,
                "smoke_failure_count": 0,
                "runtime_smoke_failure_count": 0,
                "base_runtime_smoke_failure_count": 0,
                "contract_dispatch_smoke_failure_count": 0,
                "commands_hygiene_failure_count": 0,
                "contract_hash_mismatch_count": 0,
            },
        },
    )

    payload = src.build_payload(tmp_path)
    weak_ids = {row["weak_point_id"] for row in payload["weak_points"]}

    assert "command_surface_freshness" not in weak_ids


def test_adversarial_drill_runs_only_safe_default_probes(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], _project_root: Path, _timeout_sec: int) -> dict:
        calls.append(cmd)
        assert cmd[0] == "./scripts/ops/opsctl.sh"
        assert not any(part in {"start-live", "clear-all-halts", "operator-release", "token-refresh-interactive"} for part in cmd)
        return {"cmd": cmd, "rc": 0, "payload": {"overall_status": "ready", "ok": True}, "stdout_tail": "", "stderr_tail": ""}

    payload = src.build_payload(tmp_path, run_probes=True, runner=runner)
    subcommands = [cmd[1] for cmd in calls]

    assert payload["probe_count"] == len(calls) == 8
    assert "health-fast" in subcommands
    assert "system-drift-guard" in subcommands
    assert "architecture-upgrade-scoreboard" in subcommands


def test_adversarial_drill_emits_degradation_repair_packets(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "source_verification_latest.json",
        {
            "ok": True,
            "sources": [
                {"source_id": "options_context_mesh"},
                {"source_id": "fx_market_context"},
                {"source_id": "crypto_market_context"},
            ],
        },
    )
    _write_json(
        health / "schwab_account_snapshot_refresh_latest.json",
        {
            "ok": True,
            "account_snapshot_proof": {"account_snapshot_proof_ok": True},
            "broker_truth_reconcile_v2": {"truth_score": 0.94, "truth_grade": "A"},
        },
    )
    _write_json(
        health / "training_labeling_intelligence_latest.json",
        {
            "ok": True,
            "free_label_source_enrichment": {
                "classification_counts": {"free_public_or_verified_proxy": 3},
                "materialization_ready_context_count": 3,
            },
        },
    )
    _write_json(
        health / "paper_execution_truth_layer_latest.json",
        {
            "ok": True,
            "gates": {"paper_broker_truth_reconciliation": {"status": "ready"}},
        },
    )

    payload = src.build_payload(tmp_path)
    scenarios = {row["scenario_id"]: row for row in payload["degradation_repair_packets"]}

    assert payload["degradation_scenario_count"] == 6
    assert payload["degradation_scenarios_covered"] == 6
    assert scenarios["empty_schwab_snapshot"]["repair_packet"]["action"] == "reject_empty_snapshot_and_refresh_connected_account_aggregate"
    assert scenarios["paper_activity_without_truth"]["status"] == "covered"
