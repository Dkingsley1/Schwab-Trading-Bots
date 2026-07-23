import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import bot_needs_intelligence as src


def test_bot_needs_uses_registry_data_collection_observation_count(tmp_path: Path) -> None:
    bot_id = "brain_refinery_v100_stock_crypto_overlap_context"
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text("{}", encoding="utf-8")

    record = src._classify_bot(
        {
            "bot_id": bot_id,
            "bot_role": "signal_sub_bot",
            "active": True,
            "training_excluded": True,
            "data_collection_active": True,
            "lifecycle_state": "data_collection_only",
            "minimum_training_observations": 1000,
            "data_collection_observations": 530,
            "collected_observation_count": 530,
        },
        label_row={},
        quality_row={},
        memberships={},
        walk_forward={},
        diagnostic={},
        diagnostic_path=diagnostic_path,
        calibration_override={},
        min_runs=12,
        now=datetime.now(timezone.utc),
    )

    assert record["evidence"]["observation_count"] == 530
    assert any(
        need["key"] == "collect_more_data"
        and need["summary"] == "Collect 470 more observations to reach the 1000 training floor."
        for need in record["all_needs"]
    )


def test_bot_needs_materializes_label_depth_when_raw_observations_exist(tmp_path: Path) -> None:
    bot_id = "brain_refinery_v188_host_resource_pressure_guard"
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text("{}", encoding="utf-8")

    record = src._classify_bot(
        {
            "bot_id": bot_id,
            "bot_role": "infrastructure_sub_bot",
            "active": True,
            "training_excluded": True,
            "data_collection_active": True,
            "lifecycle_state": "data_collection_only",
            "minimum_training_observations": 1000,
            "data_collection_observations": 1400,
        },
        label_row={"recommendation": "materialize_label_depth"},
        quality_row={},
        memberships={},
        walk_forward={},
        diagnostic={
            "sample_count": 0,
            "eligible_sequences": 0,
            "observation_count": 1400,
            "runtime_meta": {
                "label_depth_contract": {
                    "status": "materialize_label_depth",
                    "estimated_usable_sample_capacity": 200,
                    "usable_sample_gap": 0,
                    "next_action": "materialize_point_in_time_label_depth_from_existing_observations",
                }
            },
        },
        diagnostic_path=diagnostic_path,
        calibration_override={},
        min_runs=12,
        now=datetime.now(timezone.utc),
    )

    assert record["primary_need"] == "materialize_label_depth"
    assert record["next_command"] == [
        "./scripts/ops/opsctl.sh",
        "training-data-intake",
        "--apply",
        "--include-bot-ids",
        bot_id,
        "--json",
    ]
    assert record["effectiveness_prescription"]["can_train_now"] is False
    assert record["evidence"]["estimated_usable_sample_capacity"] == 200


def test_bot_needs_refreshes_diagnostics_when_label_depth_capacity_is_ready(tmp_path: Path) -> None:
    bot_id = "brain_refinery_v314_collection_coverage_gap_mapper"
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text("{}", encoding="utf-8")

    record = src._classify_bot(
        {
            "bot_id": bot_id,
            "bot_role": "infrastructure_sub_bot",
            "active": True,
            "training_excluded": False,
            "data_collection_active": True,
            "lifecycle_state": "data_collection_only",
            "minimum_training_observations": 1000,
            "data_collection_observations": 1400,
        },
        label_row={},
        quality_row={},
        memberships={},
        walk_forward={},
        diagnostic={
            "sample_count": 0,
            "eligible_sequences": 0,
            "observation_count": 1400,
            "runtime_meta": {
                "label_depth_contract": {
                    "status": "label_depth_ready_for_real_diagnostic_refresh",
                    "estimated_usable_sample_capacity": 200,
                    "usable_sample_gap": 0,
                    "next_action": "refresh_with_real_samples",
                }
            },
        },
        diagnostic_path=diagnostic_path,
        calibration_override={},
        min_runs=12,
        now=datetime.now(timezone.utc),
    )

    assert record["primary_need"] == "refresh_training_diagnostics"
    assert record["next_command"] == ["./scripts/ops/opsctl.sh", "bot-quality-autopilot", "--apply", "--json"]


def test_bot_needs_routes_deleted_sample_starved_to_collection_requalification(tmp_path: Path) -> None:
    bot_id = "brain_refinery_v23_atr_adx_regime"
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text("{}", encoding="utf-8")

    record = src._classify_bot(
        {
            "bot_id": bot_id,
            "bot_role": "signal_sub_bot",
            "active": False,
            "deleted_from_rotation": True,
            "training_excluded": False,
            "data_collection_active": False,
            "lifecycle_state": "deleted",
        },
        label_row={},
        quality_row={},
        memberships={},
        walk_forward={},
        diagnostic={
            "status": "deferred_sample_starved",
            "sample_count": 0,
            "eligible_sequences": 0,
            "observation_count": 6,
            "sequence_count": 3,
        },
        diagnostic_path=diagnostic_path,
        calibration_override={},
        min_runs=12,
        now=datetime.now(timezone.utc),
    )

    assert record["primary_need"] == "reactivate_sample_starved_collection"
    assert record["effectiveness_prescription"]["can_train_now"] is False
    assert record["next_command"] == [
        "./scripts/ops/opsctl.sh",
        "training-requalification",
        "--include-bot-ids",
        bot_id,
        "--include-sample-starved-deleted",
        "--apply-repair",
        "--write-queue",
        "--json",
    ]


def test_bot_needs_surfaces_overfitting_as_actionable_blocker(tmp_path: Path) -> None:
    bot_id = "brain_refinery_v45_intraday_open_close_regimes"
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text("{}", encoding="utf-8")

    record = src._classify_bot(
        {
            "bot_id": bot_id,
            "bot_role": "signal_sub_bot",
            "active": True,
            "data_collection_active": True,
            "lifecycle_state": "active",
        },
        label_row={},
        quality_row={},
        memberships={},
        walk_forward={"runs": 12, "status": "pass"},
        diagnostic={},
        diagnostic_path=diagnostic_path,
        calibration_override={},
        min_runs=12,
        now=datetime.now(timezone.utc),
        overfit_row={
            "bot_id": bot_id,
            "status": "severe_overfit",
            "risk_score": 0.88,
            "train_forward_gap": 0.17,
            "policy": {"may_teach": False, "may_promote": False},
        },
    )

    assert record["primary_need"] == "reduce_overfitting"
    assert record["next_command"] == ["./scripts/ops/opsctl.sh", "overfitting-awareness", "--json"]
    assert record["evidence"]["overfit_status"] == "severe_overfit"
    assert record["effectiveness_prescription"]["can_train_now"] is False


def test_balanced_signal_bot_gets_long_precision_assignment(tmp_path: Path) -> None:
    bot_id = "brain_refinery_v47_swing_1w_3w"
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text("{}", encoding="utf-8")

    record = src._classify_bot(
        {
            "bot_id": bot_id,
            "bot_role": "signal_sub_bot",
            "active": True,
            "data_collection_active": True,
            "lifecycle_state": "active",
        },
        label_row={},
        quality_row={},
        memberships={},
        walk_forward={"runs": 12, "status": "pass"},
        diagnostic={
            "sample_count": 420,
            "positive_rate": 0.49,
            "metrics": {
                "acted_coverage": 0.44,
                "acted_accuracy": 0.55,
                "long_precision": 0.0,
                "short_precision": 0.56,
            },
        },
        diagnostic_path=diagnostic_path,
        calibration_override={},
        min_runs=12,
        now=datetime.now(timezone.utc),
    )

    assert record["primary_need"] == "repair_long_precision"
    assert record["next_command"] == ["./scripts/ops/opsctl.sh", "calibration-control", "--apply", "--json"]
    assert record["evidence"]["precision_contract"]["type"] == "balanced_directional"
    assert record["evidence"]["precision_contract"]["required_sides"] == ["long", "short"]
    assert record["evidence"]["precision_gaps"]["long_precision_gap"] > 0


def test_bot_needs_uses_fresh_repaired_diagnostic_file_age(tmp_path: Path) -> None:
    bot_id = "brain_refinery_v47_swing_1w_3w"
    now = datetime.now(timezone.utc)
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text("{}", encoding="utf-8")
    os.utime(diagnostic_path, (now.timestamp(), now.timestamp()))

    record = src._classify_bot(
        {
            "bot_id": bot_id,
            "bot_role": "signal_sub_bot",
            "active": True,
            "data_collection_active": True,
            "lifecycle_state": "active",
        },
        label_row={"diagnostic_age_hours": 96.0},
        quality_row={},
        memberships={},
        walk_forward={"runs": 12, "status": "pass"},
        diagnostic={
            "timestamp_utc": (now - timedelta(hours=96)).isoformat(),
            "sample_count": 420,
            "positive_rate": 0.49,
            "metrics": {
                "acted_coverage": 0.44,
                "acted_accuracy": 0.55,
                "long_precision": 0.0,
                "short_precision": 0.56,
            },
        },
        diagnostic_path=diagnostic_path,
        calibration_override={},
        min_runs=12,
        now=now,
    )

    need_keys = {need["key"] for need in record["all_needs"]}
    assert "refresh_training_diagnostics" not in need_keys
    assert record["evidence"]["diagnostic_age_hours"] == 0.0
    assert record["evidence"]["diagnostic_payload_age_hours"] >= 95.0
    assert record["evidence"]["diagnostic_label_age_hours"] == 96.0
    assert record["evidence"]["diagnostic_file_age_hours"] == 0.0


def test_defensive_bot_does_not_require_long_precision_first(tmp_path: Path) -> None:
    bot_id = "brain_refinery_v31_defensive_rotation"
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text("{}", encoding="utf-8")

    record = src._classify_bot(
        {
            "bot_id": bot_id,
            "bot_role": "options_sub_bot",
            "active": True,
            "data_collection_active": True,
            "lifecycle_state": "active",
        },
        label_row={},
        quality_row={},
        memberships={},
        walk_forward={"runs": 12, "status": "pass"},
        diagnostic={
            "sample_count": 420,
            "positive_rate": 0.52,
            "metrics": {
                "acted_coverage": 0.36,
                "acted_accuracy": 0.54,
                "long_precision": 0.0,
                "short_precision": 0.56,
            },
        },
        diagnostic_path=diagnostic_path,
        calibration_override={},
        min_runs=12,
        now=datetime.now(timezone.utc),
    )

    need_keys = {need["key"] for need in record["all_needs"]}
    assert "repair_long_precision" not in need_keys
    assert record["primary_need"] == "monitor"
    assert record["evidence"]["precision_contract"]["type"] == "defensive_or_short_bias"
    assert record["evidence"]["precision_contract"]["required_sides"] == ["short"]


def test_infrastructure_bot_gets_guard_precision_assignment(tmp_path: Path) -> None:
    bot_id = "brain_refinery_v80_execution_feasibility_sentinel"
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text("{}", encoding="utf-8")

    record = src._classify_bot(
        {
            "bot_id": bot_id,
            "bot_role": "infrastructure_sub_bot",
            "active": True,
            "data_collection_active": True,
            "lifecycle_state": "active",
        },
        label_row={},
        quality_row={},
        memberships={},
        walk_forward={"runs": 12, "status": "pass"},
        diagnostic={
            "sample_count": 420,
            "positive_rate": 0.48,
            "metrics": {
                "acted_coverage": 0.86,
                "acted_accuracy": 0.49,
                "long_precision": 0.0,
                "short_precision": 0.58,
            },
        },
        diagnostic_path=diagnostic_path,
        calibration_override={},
        min_runs=12,
        now=datetime.now(timezone.utc),
    )

    need_keys = {need["key"] for need in record["all_needs"]}
    assert "repair_long_precision" not in need_keys
    assert record["primary_need"] == "repair_guard_false_positive_control"
    assert record["next_command"] == ["./scripts/ops/opsctl.sh", "calibration-control", "--apply", "--json"]
    assert record["evidence"]["precision_contract"]["type"] == "guard_control"


def test_bot_needs_training_selector_and_zero_observation_repair_contract() -> None:
    ready_record = {
        "bot_id": "brain_refinery_v50_investment_drawdown_risk",
        "active": True,
        "data_collection_active": True,
        "primary_need": "targeted_quality_retrain",
        "priority": 84.0,
        "effectiveness_prescription": {"can_train_now": True},
        "evidence": {
            "sample_count": 420,
            "observation_count": 1200,
            "positive_rate": 0.48,
            "quality_score": 0.92,
            "test_accuracy": 0.57,
            "walk_forward_runs_remaining": 1,
            "overfit_status": "generalization_clean",
        },
    }
    zero_record = {
        "bot_id": "brain_refinery_v999_collection_probe",
        "active": True,
        "data_collection_active": True,
        "primary_need": "collect_more_data",
        "next_command": ["./scripts/ops/opsctl.sh", "training-label-audit", "--json"],
        "effectiveness_prescription": {"can_train_now": False},
        "evidence": {"sample_count": 0, "observation_count": 0, "positive_rate": 0.5},
    }
    expected_training_labeling_observer = {
        "bot_id": "brain_refinery_v1661_training_labeling_label_contract_normalizer_telemetry_collector_bot",
        "active": True,
        "training_excluded": True,
        "data_collection_active": True,
        "lifecycle_state": "data_collection_only",
        "primary_need": "collect_more_data",
        "next_command": ["./scripts/ops/opsctl.sh", "training-label-audit", "--json"],
        "effectiveness_prescription": {"can_train_now": False},
        "evidence": {"sample_count": 0, "observation_count": 0, "positive_rate": 0.5},
    }

    selector = src._training_candidate_selector([ready_record, zero_record, expected_training_labeling_observer])
    repair = src._zero_observation_repair_contract([ready_record, zero_record, expected_training_labeling_observer])

    assert selector["mode"] == "training_candidate_selector_v2"
    assert selector["selected_candidates"][0]["bot_id"] == ready_record["bot_id"]
    assert selector["recommended_batch_command"][3] == ready_record["bot_id"]
    assert repair["mode"] == "zero_observation_collector_repair_v2"
    assert repair["zero_observation_count"] == 1
    assert repair["zero_observation_bots"][0]["bot_id"] == zero_record["bot_id"]
    assert repair["excluded_expected_observer_count"] == 1
    assert repair["excluded_expected_observer_bots"][0]["bot_id"] == expected_training_labeling_observer["bot_id"]
