from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from core import indicator_bot_common as common


def test_flatten_and_load_model_round_trip(tmp_path: Path) -> None:
    src = common.TradingBrain(8)
    mx.eval(src.parameters())
    flat = common._flatten_param_tree(src.parameters())

    assert "layer1.weight" in flat
    assert "out.bias" in flat

    model_path = tmp_path / "teacher.npz"
    np.savez(model_path, **flat)

    restored = common.TradingBrain(8)
    common.load_model(restored, str(model_path))
    mx.eval(restored.parameters())

    restored_flat = common._flatten_param_tree(restored.parameters())
    for key, value in flat.items():
        np.testing.assert_allclose(restored_flat[key], value)


def test_snapshot_and_restore_model_round_trip() -> None:
    model = common.TradingBrain(8)
    mx.eval(model.parameters())
    baseline = common._snapshot_model_params(model)

    mutated = {key: (value + 1.0) for key, value in baseline.items()}
    common._assign_param_tree(model.parameters(), mutated)
    mx.eval(model.parameters())

    common._restore_model_params(model, baseline)
    restored = common._flatten_param_tree(model.parameters())
    for key, value in baseline.items():
        np.testing.assert_allclose(restored[key], value)


def test_teacher_soft_targets_align_to_student_anchors(monkeypatch) -> None:
    panel = common.simulate_market_panel(n=32)

    def fake_teacher_spec(project_root: Path, bot_id: str):
        return {
            "bot_id": bot_id,
            "model_path": Path("/tmp/fake_teacher.npz"),
            "config": {"window": 3, "horizon": 1, "input_dim": 1},
            "feature_builder": lambda panel: np.zeros((32, 1), dtype=np.float32),
        }

    def fake_make_windowed_dataset(features, close, window, horizon, *, return_anchor_index=False):
        x = mx.array(np.array([[0.0], [1.0], [2.0]], dtype=np.float32))
        y = mx.array(np.array([[0.0], [1.0], [1.0]], dtype=np.float32))
        anchors = np.array([10, 12, 15], dtype=np.int64)
        if return_anchor_index:
            return x, y, anchors
        return x, y

    class FakeTeacher:
        def __init__(self, input_dim: int):
            self.input_dim = input_dim

        def __call__(self, x):
            return x

    monkeypatch.setattr(common, "_load_teacher_spec", fake_teacher_spec)
    monkeypatch.setattr(common, "make_windowed_dataset", fake_make_windowed_dataset)
    monkeypatch.setattr(common, "TradingBrain", FakeTeacher)
    monkeypatch.setattr(common, "load_model", lambda model, path: model)

    soft, used = common._teacher_soft_targets(
        project_root=Path("/tmp"),
        teacher_ids=["brain_refinery_v10_seasonal"],
        panel=panel,
        prices=panel["close"],
        student_anchor_idx=np.array([10, 11, 12, 15], dtype=np.int64),
    )

    assert used == ["brain_refinery_v10_seasonal"]
    assert soft is not None
    np.testing.assert_allclose(soft[[0, 2, 3]], np.array([0.5, 0.7310586, 0.8807971], dtype=np.float32), rtol=1e-5)
    assert np.isnan(soft[1])
