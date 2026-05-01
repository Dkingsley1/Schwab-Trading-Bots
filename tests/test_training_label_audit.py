import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.training_label_audit as audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_build_label_audit_payload_surfaces_filter_and_abstention_actions(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    diagnostics_dir = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v4_simple", "bot_role": "signal_sub_bot", "active": True},
                {"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy", "bot_role": "signal_sub_bot", "active": True},
                {"bot_id": "brain_refinery_v75_model_drift_guard", "bot_role": "signal_sub_bot", "active": True},
            ]
        },
    )
    _write_json(
        diagnostics_dir / "brain_refinery_v4_simple_latest.json",
        {
            "status": "deferred_sample_starved",
            "sample_count": 0,
            "eligible_sequences": 14,
            "sequence_count": 14,
            "observation_count": 180,
            "positive_rate": 0.41,
            "skipped_filtered": 18,
            "skipped_low_confidence": 3,
            "skipped_labels": 1,
        },
    )
    _write_json(
        diagnostics_dir / "brain_refinery_v43_intraday_ultrafast_proxy_latest.json",
        {
            "status": "failed",
            "sample_count": 420,
            "positive_rate": 0.49,
            "metrics": {
                "acted_coverage": 0.74,
                "acted_accuracy": 0.51,
                "accuracy_lift_over_majority": -0.08,
                "long_precision": 0.54,
                "short_precision": 0.52,
                "label_balance_score": 0.51,
            },
        },
    )
    _write_json(
        diagnostics_dir / "brain_refinery_v75_model_drift_guard_latest.json",
        {
            "status": "failed",
            "sample_count": 280,
            "positive_rate": 0.52,
            "metrics": {
                "acted_coverage": 0.18,
                "acted_accuracy": 0.58,
                "accuracy_lift_over_majority": 0.03,
                "long_precision": 0.67,
                "short_precision": 0.41,
                "label_balance_score": 0.46,
            },
        },
    )

    payload = audit.build_label_audit_payload(
        registry_path=registry_path,
        diagnostics_dir=diagnostics_dir,
    )

    assert payload["active_rows"] == 3
    assert payload["recommendation_counts"]["relax_sample_filter"] == 1
    assert payload["recommendation_counts"]["tighten_abstention_thresholds"] == 1
    assert payload["recommendation_counts"]["use_side_specific_thresholds"] == 1
    assert payload["top_actions"] == [
        "relax_sample_filter",
        "tighten_abstention_thresholds",
        "use_side_specific_thresholds",
    ]
    assert payload["active_zero_sample"][0]["bot_id"] == "brain_refinery_v4_simple"
    assert payload["active_overacting"][0]["bot_id"] == "brain_refinery_v43_intraday_ultrafast_proxy"


def test_build_label_audit_marks_missing_diagnostics_for_refresh(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    diagnostics_dir = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v17_mixed_regime", "bot_role": "signal_sub_bot", "active": True},
            ]
        },
    )

    payload = audit.build_label_audit_payload(
        registry_path=registry_path,
        diagnostics_dir=diagnostics_dir,
    )

    assert payload["recommendation_counts"]["refresh_training_diagnostics"] == 1
    assert payload["top_actions"] == ["refresh_training_diagnostics"]
    assert payload["active_zero_sample"][0]["bot_id"] == "brain_refinery_v17_mixed_regime"


def test_collect_only_bot_gets_collection_diagnostics_and_label_contract(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    diagnostics_dir = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v247_market_neutral_pairs_execution_bot",
                    "bot_role": "signal_sub_bot",
                    "slot_kind": "market_neutral_signal",
                    "lifecycle_state": "data_collection_only",
                    "training_excluded": True,
                    "data_collection_active": True,
                    "active": True,
                },
            ]
        },
    )

    payload = audit.build_label_audit_payload(
        registry_path=registry_path,
        diagnostics_dir=diagnostics_dir,
    )

    row = payload["active_zero_sample"][0]
    assert payload["recommendation_counts"]["create_collect_only_diagnostics"] == 1
    assert payload["top_actions"] == ["create_collect_only_diagnostics"]
    assert row["label_family"] == "spread_convergence"
    assert row["primary_label_horizon"] == "spread_zscore_reversion_3d"
    assert "correlation_matrix" in row["required_label_context"]


def test_monitorable_bot_without_label_contract_gets_upgrade_action(tmp_path: Path) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    diagnostics_dir = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        registry_path,
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v252_position_lifecycle_trim_add_hold_bot",
                    "bot_role": "signal_sub_bot",
                    "slot_kind": "position_lifecycle_signal",
                    "active": True,
                },
            ]
        },
    )
    _write_json(
        diagnostics_dir / "brain_refinery_v252_position_lifecycle_trim_add_hold_bot_latest.json",
        {
            "status": "ok",
            "sample_count": 420,
            "positive_rate": 0.50,
            "metrics": {
                "acted_coverage": 0.20,
                "acted_accuracy": 0.59,
                "accuracy_lift_over_majority": 0.04,
                "long_precision": 0.58,
                "short_precision": 0.55,
                "label_balance_score": 0.50,
            },
            "runtime_meta": {"sample_count": 420},
        },
    )

    payload = audit.build_label_audit_payload(
        registry_path=registry_path,
        diagnostics_dir=diagnostics_dir,
    )

    assert payload["recommendation_counts"]["upgrade_label_contract"] == 1
    assert payload["top_actions"] == ["upgrade_label_contract"]
    assert payload["active_label_contract_upgrades"][0]["label_family"] == "position_management"
