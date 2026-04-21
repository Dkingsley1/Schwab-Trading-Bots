import json
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


def test_resolve_training_guard_profile_applies_intraday_defaults() -> None:
    profile = common._resolve_training_guard_profile(
        "brain_refinery_v43_intraday_ultrafast_proxy",
        min_label_balance_score=None,
        min_acted_coverage=None,
        max_acted_coverage=None,
    )

    assert profile["family"] == "intraday"
    assert profile["min_label_balance_score"] >= 0.22
    assert profile["min_acted_coverage"] >= 0.03
    assert profile["max_acted_coverage"] <= 0.48


def test_paper_loss_hard_negative_context_reads_latest_report(tmp_path: Path) -> None:
    report_path = tmp_path / "governance" / "health" / "paper_performance_latest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "sleeve_latest": [
                    {
                        "profile": "intraday_aggressive",
                        "ending_net_pnl_total": -12.5,
                        "win_rate": 0.25,
                        "top_losing_strategies": [
                            {
                                "strategy": "paper_mirror::brain_refinery_v43_intraday_ultrafast_proxy",
                                "ending_net_pnl_total": -3.4,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    context = common._paper_loss_hard_negative_context(tmp_path, "brain_refinery_v43_intraday_ultrafast_proxy")

    assert context["enabled"] is True
    assert context["matched_profiles"] == ["intraday_aggressive"]
    assert context["loss_score"] == 3.4
    assert context["weight_multiplier"] > 1.0


def test_runtime_training_autofix_plan_expands_scope_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_TRAIN_AUTOFIX_INSUFFICIENT_DATA", "1")
    plan = common._runtime_training_autofix_plan(
        lookback_days=14,
        symbol_allowlist=["AAPL", "MSFT"],
        min_confidence=0.35,
        sample_stride=3,
    )

    assert plan[0]["reason"] == "base"
    assert any(row["reason"] == "widen_lookback" for row in plan)
    assert any(row["reason"] == "broaden_symbol_scope" and row["symbol_allowlist"] == [] for row in plan)
    assert any(int(row["sample_stride"]) == 1 for row in plan)


def test_require_mlx_runtime_message_mentions_portable_mode(monkeypatch) -> None:
    monkeypatch.setattr(common, "_MLX_AVAILABLE", False)
    monkeypatch.setattr(common, "_MLX_IMPORT_ERROR", ModuleNotFoundError("mlx unavailable"))
    monkeypatch.setenv("BOT_RUNTIME_ACCESS_MODE", "portable")
    monkeypatch.setenv("BOT_ML_RUNTIME_OPTIONAL", "1")
    monkeypatch.setenv("BOT_ML_BACKEND", "portable_auto")

    try:
        common._require_mlx_runtime("runtime training")
    except RuntimeError as exc:
        text = str(exc)
    else:
        raise AssertionError("expected RuntimeError when MLX is unavailable")

    assert "portable mode" in text
    assert "backend=portable_auto" in text
    assert "runtime training" in text


def test_resolve_training_guard_profile_adapts_to_weak_live_behavior(tmp_path: Path) -> None:
    report_path = tmp_path / "governance" / "health" / "paper_performance_latest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "sleeve_latest": [
                    {
                        "profile": "intraday_aggressive",
                        "ending_net_pnl_total": -21.0,
                        "win_rate": 0.20,
                        "losing_strategy_count": 7,
                        "winning_strategy_count": 1,
                        "top_losing_strategies": [
                            {
                                "strategy": "paper_mirror::brain_refinery_v43_intraday_ultrafast_proxy",
                                "ending_net_pnl_total": -5.2,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    profile = common._resolve_training_guard_profile(
        "brain_refinery_v43_intraday_ultrafast_proxy",
        min_label_balance_score=None,
        min_acted_coverage=None,
        max_acted_coverage=None,
        project_root=tmp_path,
    )

    assert profile["adaptive_from_live_behavior"]["weak_profile_count"] == 1
    assert profile["min_label_balance_score"] > 0.22
    assert profile["max_acted_coverage"] < 0.48


def test_resolve_training_guard_profile_bond_family_adapts_with_weaker_coverage(tmp_path: Path) -> None:
    report_path = tmp_path / "governance" / "health" / "paper_performance_latest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "sleeve_latest": [
                    {
                        "profile": "bond",
                        "ending_net_pnl_total": -18.0,
                        "win_rate": 0.30,
                        "losing_strategy_count": 4,
                        "winning_strategy_count": 1,
                        "top_losing_strategies": [
                            {
                                "strategy": "paper_mirror::brain_refinery_v95_rates_regime_bond_bot",
                                "ending_net_pnl_total": -4.0,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    profile = common._resolve_training_guard_profile(
        "brain_refinery_v95_rates_regime_bond_bot",
        min_label_balance_score=None,
        min_acted_coverage=None,
        max_acted_coverage=None,
        project_root=tmp_path,
    )

    assert profile["family"] == "bond"
    assert profile["min_label_balance_score"] > 0.15
    assert profile["min_acted_coverage"] >= 0.02


def test_resolve_learned_acted_threshold_applies_family_and_bot_overrides(tmp_path: Path) -> None:
    override_path = tmp_path / "governance" / "health" / "calibration_abstention_overrides_latest.json"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(
        json.dumps(
            {
                "family_overrides": {
                    "dividend": {
                        "mode": "tighten",
                        "acted_prob_threshold_uplift": 0.02,
                    }
                },
                "bot_overrides": {
                    "brain_refinery_v99_defensive_dividend_concentration": {
                        "mode": "tighten",
                        "acted_prob_threshold_uplift": 0.03,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    threshold, meta = common._resolve_learned_acted_threshold(
        tmp_path,
        run_tag="brain_refinery_v99_defensive_dividend_concentration",
        family="dividend",
        base_threshold=0.65,
    )

    assert abs(threshold - 0.70) < 1e-9
    assert len(meta["applied_sources"]) == 2
    assert meta["applied_sources"][0]["scope"] == "family"
    assert meta["applied_sources"][1]["scope"] == "bot"


def test_resolve_runtime_training_path_profile_relaxes_sample_starved_intraday_bot(tmp_path: Path) -> None:
    (tmp_path / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                        "bot_role": "signal_sub_bot",
                        "active": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    diagnostics_dir = tmp_path / "governance" / "training_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / "brain_refinery_v43_intraday_ultrafast_proxy_latest.json").write_text(
        json.dumps(
            {
                "status": "deferred_sample_starved",
                "sample_count": 0,
                "eligible_sequences": 0,
                "sequence_count": 0,
                "observation_count": 0,
                "positive_rate": 0.0,
                "skipped_low_confidence": 8,
            }
        ),
        encoding="utf-8",
    )

    profile = common._resolve_runtime_training_path_profile(
        "brain_refinery_v43_intraday_ultrafast_proxy",
        lookback_days=14,
        sample_stride=3,
        min_confidence=0.35,
        batch_size=128,
        patience=18,
        epochs=200,
        min_samples=256,
        min_sequences=4,
        min_positive_samples=40,
        min_negative_samples=40,
        project_root=tmp_path,
    )

    assert profile["family"] == "intraday"
    assert profile["diagnostic_adaptation"]["status"] == "deferred_sample_starved"
    assert profile["lookback_days"] == 60
    assert profile["sample_stride"] == 2
    assert abs(profile["min_confidence"] - 0.32) < 1e-9
    assert profile["batch_size"] == 96
    assert profile["min_samples"] == 192
    assert profile["min_sequences"] == 3
    assert profile["min_positive_samples"] == 26
    assert profile["min_negative_samples"] == 26
    assert profile["autofix_max_lookback_days"] >= 120
    assert abs(profile["autofix_min_confidence_floor"] - 0.18) < 1e-9


def test_resolve_runtime_training_path_profile_uses_infrastructure_role_overlay(tmp_path: Path) -> None:
    (tmp_path / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v101_infra_guard",
                        "bot_role": "infrastructure_sub_bot",
                        "active": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    profile = common._resolve_runtime_training_path_profile(
        "brain_refinery_v101_infra_guard",
        lookback_days=21,
        sample_stride=3,
        min_confidence=0.45,
        batch_size=128,
        patience=18,
        epochs=200,
        min_samples=220,
        min_sequences=4,
        min_positive_samples=28,
        min_negative_samples=28,
        project_root=tmp_path,
    )

    assert profile["family"] == "core"
    assert profile["bot_role"] == "infrastructure_sub_bot"
    assert profile["lookback_days"] == 89
    assert profile["sample_stride"] == 2
    assert abs(profile["min_confidence"] - 0.33) < 1e-9
    assert profile["batch_size"] == 96
    assert profile["min_samples"] == 144
    assert profile["min_positive_samples"] == 14
    assert profile["min_negative_samples"] == 14


def test_resolve_runtime_training_path_profile_applies_memory_caps(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                        "bot_role": "signal_sub_bot",
                        "active": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BOT_MEMORY_EFFICIENCY_PROFILE", "constrained")
    monkeypatch.setenv("RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR", "3")
    monkeypatch.setenv("RUNTIME_TRAIN_BATCH_SIZE_CAP", "48")
    monkeypatch.setenv("RUNTIME_TRAIN_MAX_SAMPLES", "8000")

    profile = common._resolve_runtime_training_path_profile(
        "brain_refinery_v43_intraday_ultrafast_proxy",
        lookback_days=14,
        sample_stride=1,
        min_confidence=0.35,
        batch_size=128,
        patience=18,
        epochs=200,
        min_samples=256,
        min_sequences=4,
        min_positive_samples=40,
        min_negative_samples=40,
        project_root=tmp_path,
    )

    assert profile["sample_stride"] == 2
    assert profile["batch_size"] == 48
    assert profile["max_samples"] == 8000
    assert profile["memory_efficiency"]["profile"] == "constrained"
    assert profile["memory_efficiency"]["batch_size_cap"] == 48


def test_resolve_runtime_training_path_profile_coverage_canary_can_override_memory_stride_floor(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                        "bot_role": "signal_sub_bot",
                        "active": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BOT_MEMORY_EFFICIENCY_PROFILE", "constrained")
    monkeypatch.setenv("RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR", "3")
    monkeypatch.setenv("RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE", "1")
    monkeypatch.setenv("RETRAIN_PROFILE", "coverage_canary")

    profile = common._resolve_runtime_training_path_profile(
        "brain_refinery_v43_intraday_ultrafast_proxy",
        lookback_days=14,
        sample_stride=1,
        min_confidence=0.35,
        batch_size=128,
        patience=18,
        epochs=200,
        min_samples=256,
        min_sequences=4,
        min_positive_samples=40,
        min_negative_samples=40,
        project_root=tmp_path,
    )

    assert profile["sample_stride"] == 1
    assert profile["memory_efficiency"]["sample_stride_floor"] == 1


def test_resolve_runtime_training_path_profile_honors_lookback_cap(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v99_defensive_dividend_concentration",
                        "bot_role": "options_sub_bot",
                        "active": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RUNTIME_TRAIN_LOOKBACK_DAYS_CAP", "45")

    profile = common._resolve_runtime_training_path_profile(
        "brain_refinery_v99_defensive_dividend_concentration",
        lookback_days=45,
        sample_stride=1,
        min_confidence=0.40,
        batch_size=128,
        patience=18,
        epochs=220,
        min_samples=224,
        min_sequences=6,
        min_positive_samples=0,
        min_negative_samples=0,
        project_root=tmp_path,
    )

    assert profile["bot_role"] == "options_sub_bot"
    assert profile["lookback_days"] == 45
    assert profile["explicit_adjustments"]["lookback_days_capped"] is True
    assert profile["memory_efficiency"]["lookback_cap"] == 45


def test_train_runtime_indicator_bot_applies_training_path_before_autofix(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        common,
        "_resolve_runtime_training_path_profile",
        lambda *args, **kwargs: {
            "family": "intraday",
            "bot_role": "signal_sub_bot",
            "lookback_days": 90,
            "sample_stride": 1,
            "min_confidence": 0.12,
            "batch_size": 64,
            "patience": 24,
            "epochs": 240,
            "min_samples": 120,
            "min_sequences": 3,
            "min_positive_samples": 8,
            "min_negative_samples": 9,
            "max_samples": 9000,
            "autofix_max_lookback_days": 180,
            "autofix_min_confidence_floor": 0.06,
            "explicit_adjustments": {},
            "diagnostic_adaptation": {},
            "registry_context": {},
        },
    )
    monkeypatch.setattr(
        common,
        "_resolve_training_guard_profile",
        lambda *args, **kwargs: {
            "family": "intraday",
            "min_label_balance_score": 0.22,
            "min_acted_coverage": 0.03,
            "max_acted_coverage": 0.48,
        },
    )

    def _fake_autofix_plan(**kwargs):
        captured["autofix_plan"] = dict(kwargs)
        return [
            {
                "reason": "base",
                "lookback_days": int(kwargs["lookback_days"]),
                "symbol_allowlist": list(kwargs["symbol_allowlist"] or []),
                "min_confidence": float(kwargs["min_confidence"]),
                "sample_stride": int(kwargs["sample_stride"]),
            }
        ]

    def _fake_load_sequences(project_root, *, lookback_days, mode_allowlist, symbol_allowlist):
        captured["load_sequences"] = {
            "lookback_days": int(lookback_days),
            "mode_allowlist": list(mode_allowlist or []),
            "symbol_allowlist": list(symbol_allowlist or []),
        }
        return {}

    def _fake_make_dataset(**kwargs):
        captured["make_dataset"] = {
            "min_confidence": float(kwargs["min_confidence"]),
            "sample_stride": int(kwargs["sample_stride"]),
            "max_samples": int(kwargs["max_samples"]),
        }
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            {
                "eligible_sequences": 0,
                "positive_rate": 0.0,
                "sequence_count": 0,
                "skipped_filtered": 0,
                "skipped_low_confidence": 0,
                "skipped_labels": 0,
            },
        )

    monkeypatch.setattr(common, "_runtime_training_autofix_plan", _fake_autofix_plan)
    monkeypatch.setattr(common, "load_runtime_observation_sequences", _fake_load_sequences)
    monkeypatch.setattr(common, "make_runtime_windowed_dataset", _fake_make_dataset)

    result = common.train_runtime_indicator_bot(
        run_tag="brain_refinery_v43_intraday_ultrafast_proxy",
        feature_names=["x"],
        runtime_feature_builder=lambda sequence, index: np.zeros((1,), dtype=np.float32),
        runtime_label_builder=lambda sequence, index, horizon: 1.0,
        mode_allowlist=["intraday_aggressive"],
        symbol_allowlist=["SPY"],
        lookback_days=14,
        sample_stride=3,
        min_confidence=0.30,
        batch_size=128,
        patience=18,
        epochs=200,
        min_samples=256,
        min_sequences=4,
        min_positive_samples=20,
        min_negative_samples=20,
        fallback_trainer=lambda: "fallback",
        hard_negative_mining=False,
    )

    assert result == "fallback"
    assert captured["autofix_plan"]["lookback_days"] == 90
    assert captured["autofix_plan"]["min_confidence"] == 0.12
    assert captured["autofix_plan"]["sample_stride"] == 1
    assert captured["autofix_plan"]["max_lookback_days"] == 180
    assert captured["autofix_plan"]["min_confidence_floor"] == 0.06
    assert captured["load_sequences"]["lookback_days"] == 90
    assert captured["make_dataset"]["min_confidence"] == 0.12
    assert captured["make_dataset"]["sample_stride"] == 1
    assert captured["make_dataset"]["max_samples"] == 9000


def test_deferred_sample_starved_reason_writes_diagnostics(tmp_path: Path) -> None:
    message = common._deferred_sample_starved_reason(
        run_tag="brain_refinery_v43_intraday_ultrafast_proxy",
        project_root=tmp_path,
        sample_count=12,
        eligible_sequences=1,
        positive_rate=0.01,
        autofix_attempts=[{"reason": "full_recovery", "lookback_days": 60, "sample_stride": 1}],
    )

    latest = tmp_path / "governance" / "training_diagnostics" / "brain_refinery_v43_intraday_ultrafast_proxy_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))

    assert "defer_runtime_training_until_more_data" in message
    assert payload["status"] == "deferred_sample_starved"
    assert payload["recommended_retry"]["lookback_days"] == 60
