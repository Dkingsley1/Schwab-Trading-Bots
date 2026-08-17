import json
from pathlib import Path

from scripts.ops import legacy_bot_harmonizer as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_legacy_bot_harmonizer_preserves_dead_rows_and_activates_runtime_candidates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    registry_path = project_root / "master_bot_registry.json"
    _write_json(
        registry_path,
        {
            "summary": {},
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v1",
                    "bot_role": "signal_sub_bot",
                    "active": False,
                    "reason": "no_classification_accuracy",
                    "promotion_reason": "no_candidate_accuracy",
                    "lifecycle_state": "inactive",
                },
                {
                    "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "reason": "quality_gate_hold_prev_plus_0.005",
                    "promotion_reason": "quality_gate_hold_prev_plus_0.005",
                    "lifecycle_state": "active",
                    "weight": 0.12,
                },
                {
                    "bot_id": "brain_refinery_v100_stock_crypto_overlap_context",
                    "bot_role": "signal_sub_bot",
                    "active": False,
                    "reason": "new_runtime_candidate",
                    "promotion_reason": "new_runtime_candidate",
                    "lifecycle_state": "inactive",
                },
            ],
        },
    )
    core_dir = project_root / "core"
    core_dir.mkdir(parents=True)
    (core_dir / "brain_refinery_v107_cross_asset_master_candidate.py").write_text("# local core bot\n", encoding="utf-8")

    result = src.harmonize(project_root, registry_path=registry_path, apply=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = {row["bot_id"]: row for row in registry["sub_bots"]}

    assert result["applied"] is True
    assert result["summary"]["added_missing_registry_rows"] == 1
    assert "brain_refinery_v107_cross_asset_master_candidate" in rows
    assert rows["brain_refinery_v1"]["active"] is False
    assert rows["brain_refinery_v1"]["legacy_harmonization_version"] == src.HARMONIZATION_VERSION
    assert rows["brain_refinery_v43_intraday_ultrafast_proxy"]["active"] is True
    assert rows["brain_refinery_v43_intraday_ultrafast_proxy"]["lifecycle_state"] == "active"
    assert rows["brain_refinery_v43_intraday_ultrafast_proxy"]["weight"] == 0.12
    assert rows["brain_refinery_v100_stock_crypto_overlap_context"]["active"] is True
    assert rows["brain_refinery_v100_stock_crypto_overlap_context"]["lifecycle_state"] == "data_collection_only"
    assert rows["brain_refinery_v100_stock_crypto_overlap_context"]["training_excluded"] is True
    assert "quant_model_control" in rows["brain_refinery_v100_stock_crypto_overlap_context"]["target_functions"]
    assert "global_halt_pressure_reducer" in rows["brain_refinery_v100_stock_crypto_overlap_context"]["data_intake_collections"]
    assert "quantlib_pricing_benchmark" in rows["brain_refinery_v100_stock_crypto_overlap_context"]["data_intake_collections"]
    assert rows["brain_refinery_v107_cross_asset_master_candidate"]["active"] is True
    assert rows["brain_refinery_v107_cross_asset_master_candidate"]["lifecycle_state"] == "data_collection_only"
    assert registry["summary"]["legacy_harmonized_bots"] == 4
