import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.training_registry_audit as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_build_audit_payload_classifies_shared_and_quality_failures(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    diagnostics_dir = tmp_path / "governance" / "training_diagnostics"
    snapshot_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v4_simple",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "active",
                },
                {
                    "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "active",
                },
                {
                    "bot_id": "brain_refinery_v100_stock_crypto_overlap_context",
                    "bot_role": "signal_sub_bot",
                    "active": False,
                    "lifecycle_state": "inactive",
                    "promotion_reason": "new_runtime_candidate",
                },
            ]
        },
    )
    _write_json(
        diagnostics_dir / "brain_refinery_v4_simple_latest.json",
        {
            "status": "deferred_sample_starved",
            "sample_count": 0,
            "eligible_sequences": 0,
            "sequence_count": 0,
            "observation_count": 0,
            "positive_rate": 0.0,
            "failure_categories": ["sample_starved"],
        },
    )
    _write_json(
        diagnostics_dir / "brain_refinery_v43_intraday_ultrafast_proxy_latest.json",
        {
            "status": "failed",
            "sample_count": 420,
            "eligible_sequences": 16,
            "positive_rate": 0.51,
            "quality_failures": [
                "acted_accuracy=0.51 < min_acted_accuracy=0.54",
                "acted_coverage=0.79 > max_acted_coverage=0.26",
            ],
            "failure_categories": ["quality_guard_failure"],
        },
    )
    _write_json(
        snapshot_path,
        {
            "sequence_count": 0,
            "row_count": 0,
            "coverage": {"mode_count": 0, "symbol_count": 0},
        },
    )

    payload = src.build_audit_payload(
        registry_path=registry_path,
        diagnostics_dir=diagnostics_dir,
        snapshot_health_path=snapshot_path,
    )

    assert payload["registry_total_bots"] == 3
    assert payload["inferred_cause_counts"]["shared_runtime_input_gap"] == 1
    assert payload["inferred_cause_counts"]["quality_guard_failure"] == 1
    assert payload["inferred_cause_counts"]["new_runtime_candidate"] == 1
    assert payload["tier_counts"]["active_repair"] == 1
    assert payload["tier_counts"]["active_probation"] == 1
    assert payload["tier_counts"]["research_candidate"] == 1
    assert payload["supportability_counts"]["unsupported_runtime_inputs"] == 1
    assert payload["supportability_counts"]["supported_but_quality_failing"] == 1
    assert payload["tiers"]["active_repair"][0]["bot_id"] == "brain_refinery_v4_simple"
    assert payload["tiers"]["active_probation"][0]["bot_id"] == "brain_refinery_v43_intraday_ultrafast_proxy"
    assert payload["active_sample_starved"][0]["bot_id"] == "brain_refinery_v4_simple"
    assert payload["active_quality_failed"][0]["bot_id"] == "brain_refinery_v43_intraday_ultrafast_proxy"
    assert "rebuild_runtime_training_snapshot_and_rerun_targeted_retrain" in payload["recommendations"]
    assert "block_full_retrain_until_runtime_snapshot_has_rows" in payload["recommendations"]


def test_build_audit_payload_counts_artifact_backed_active_when_diagnostic_is_stale(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    diagnostics_dir = tmp_path / "governance" / "training_diagnostics"
    snapshot_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    model_path = tmp_path / "models" / "brain_refinery_v10_seasonal_20260416.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("model", encoding="utf-8")
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "active",
                    "model_path": str(model_path),
                    "quality_score": 0.91,
                    "test_accuracy": 0.77,
                }
            ]
        },
    )
    diag_path = diagnostics_dir / "brain_refinery_v10_seasonal_latest.json"
    _write_json(
        diag_path,
        {
            "status": "passed",
            "sample_count": 24,
            "eligible_sequences": 12,
            "sequence_count": 12,
            "positive_rate": 0.52,
        },
    )
    stale_ts = (datetime.now(timezone.utc).timestamp() - (8 * 24 * 3600))
    import os
    os.utime(diag_path, (stale_ts, stale_ts))
    _write_json(snapshot_path, {"sequence_count": 100, "row_count": 1000})

    payload = src.build_audit_payload(
        registry_path=registry_path,
        diagnostics_dir=diagnostics_dir,
        snapshot_health_path=snapshot_path,
    )

    assert payload["supportability_counts"]["artifact_backed_active"] == 1
    assert payload["supportability_counts"].get("unsupported_stale_diagnostics", 0) == 0


def test_build_audit_payload_counts_registry_seeded_active_when_snapshot_is_fresh(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    diagnostics_dir = tmp_path / "governance" / "training_diagnostics"
    snapshot_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v59_risk_sentinel",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "active",
                    "quality_score": 0.44,
                    "test_accuracy": 0.76,
                }
            ]
        },
    )
    _write_json(
        snapshot_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sequence_count": 120,
            "row_count": 1600,
        },
    )

    payload = src.build_audit_payload(
        registry_path=registry_path,
        diagnostics_dir=diagnostics_dir,
        snapshot_health_path=snapshot_path,
    )

    assert payload["runtime_snapshot_ready"] is True
    assert payload["supportability_counts"]["registry_seeded_active"] == 1
    assert payload["active_registry_seeded"][0]["bot_id"] == "brain_refinery_v59_risk_sentinel"


def test_build_audit_payload_counts_staged_support_recovery_for_bounded_stale_active(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    diagnostics_dir = tmp_path / "governance" / "training_diagnostics"
    snapshot_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v31_defensive_rotation",
                    "bot_role": "options_sub_bot",
                    "active": True,
                    "lifecycle_state": "active",
                    "reason": "min_active_floor_override_30:supportable_recovery",
                    "promotion_reason": "manual_collection_restore",
                    "quality_score": 0.19,
                    "test_accuracy": 0.75,
                    "candidate_quality_score": 0.27,
                    "candidate_test_accuracy": 0.76,
                }
            ]
        },
    )
    _write_json(
        snapshot_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sequence_count": 120,
            "row_count": 1600,
        },
    )

    payload = src.build_audit_payload(
        registry_path=registry_path,
        diagnostics_dir=diagnostics_dir,
        snapshot_health_path=snapshot_path,
    )

    assert payload["supportability_counts"]["staged_support_recovery"] == 1
    assert payload["active_staged_support_recovery"][0]["bot_id"] == "brain_refinery_v31_defensive_rotation"
