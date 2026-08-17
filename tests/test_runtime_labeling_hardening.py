from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import indicator_bot_common as indicator


def test_runtime_split_is_purged_and_strictly_chronological() -> None:
    labels = np.asarray(([1.0] * 35) + ([0.0] * 35) + ([1.0, 0.0] * 15), dtype=np.float32)

    plan = indicator._runtime_split_plan(labels, embargo_samples=4)

    train_idx = plan["train_idx"]
    val_idx = plan["val_idx"]
    test_idx = plan["test_idx"]
    assert plan["strategy"] == "purged_chronological"
    assert plan["effective_embargo_samples"] == 4
    assert plan["purged_sample_count"] == 8
    assert int(np.max(train_idx)) < int(np.min(val_idx)) < int(np.min(test_idx))
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(val_idx).isdisjoint(set(test_idx))
    assert "stratified_stats" not in plan


def test_runtime_split_does_not_move_future_class_balance_backward() -> None:
    labels = np.asarray(([1.0] * 70) + ([0.0, 1.0] * 15), dtype=np.float32)

    plan = indicator._runtime_split_plan(labels, embargo_samples=3)

    assert plan["strategy"] == "purged_chronological"
    assert "train_one_sided" in plan["split_warnings"]
    assert plan["fallback_reason"] == "future_class_balance_is_never_moved_into_earlier_splits"
    assert int(np.max(plan["train_idx"])) < int(np.min(plan["val_idx"]))


def test_registry_context_exposes_materialization_contract(tmp_path: Path) -> None:
    registry = {
        "sub_bots": [
            {
                "bot_id": "brain_refinery_v1",
                "bot_role": "signal_sub_bot",
                "active": True,
                "training_lane": "general_balanced",
                "universal_label_contract": {"label_family": "generic_directional"},
                "training_label_materialization_contract": {
                    "version": "training_label_materialization_contract_v2",
                    "objective_class": "market_outcome",
                    "directional_fallback_allowed": True,
                },
            }
        ]
    }
    (tmp_path / "master_bot_registry.json").write_text(json.dumps(registry) + "\n", encoding="utf-8")

    context = indicator._load_registry_bot_context(tmp_path, "paper_mirror::brain_refinery_v1")

    assert context["training_lane"] == "general_balanced"
    assert context["training_label_materialization_contract"]["objective_class"] == "market_outcome"
