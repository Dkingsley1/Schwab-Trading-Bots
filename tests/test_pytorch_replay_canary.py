import gzip
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import pytorch_replay_canary as src


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row) for row in rows) + "\n")


def test_example_from_row_skips_flat_pnl() -> None:
    row = {
        "action": "BUY",
        "model_score": 0.7,
        "threshold": 0.6,
        "quantity": 10,
        "realized_pnl_total": 0.0,
        "unrealized_pnl_total": 0.0,
        "metadata": {"intent_score": 0.7, "queue_depth": 3, "runtime_lane": "swing"},
    }

    assert src._example_from_row(row) is None


def test_example_from_row_uses_enriched_metadata_features() -> None:
    row = {
        "timestamp_utc": "2026-04-01T14:00:00+00:00",
        "action": "BUY",
        "model_score": 0.72,
        "threshold": 0.6,
        "quantity": 15,
        "realized_pnl_total": 2.0,
        "unrealized_pnl_total": 0.5,
        "metadata": {
            "intent_score": 0.71,
            "queue_depth": 4,
            "runtime_lane": "futures",
            "layer": "grand_master",
            "source_profile": "crypto_futures",
            "shadow_domain": "crypto",
            "lane_budget_mult": 0.7,
            "allow_live_promotion": True,
            "guard_blocked_intent": False,
            "master_weights": {"trend": 0.2, "mean_revert": 0.3, "shock": 0.5},
            "bot_weight": 0.42,
            "test_accuracy": 0.88,
            "bot_promoted": True,
            "bot_role": "signal_sub_bot",
        },
    }

    example = src._example_from_row(row)

    assert example is not None
    assert len(example["features"]) == len(src.FEATURE_NAMES)
    assert example["runtime_lane"] == "futures"
    assert example["shadow_domain"] == "crypto"
    assert example["layer"] == "grand_master"
    features = example["features"]
    assert features[src.FEATURE_NAMES.index("lane_budget_mult")] == 0.7
    assert features[src.FEATURE_NAMES.index("master_weight_shock")] == 0.5
    assert features[src.FEATURE_NAMES.index("allow_live_promotion")] == 1.0
    assert features[src.FEATURE_NAMES.index("layer_grand_master")] == 1.0
    assert features[src.FEATURE_NAMES.index("shadow_domain_crypto")] == 1.0
    assert features[src.FEATURE_NAMES.index("bot_role_signal_sub_bot")] == 1.0


def test_build_pytorch_replay_canary_runs_on_synthetic_rows(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    rows_a = []
    rows_b = []
    for idx in range(1200):
        score = 0.75 if idx % 2 == 0 else 0.35
        pnl = 1.5 if idx % 2 == 0 else -1.0
        row = {
            "timestamp_utc": f"2026-04-01T14:{idx % 60:02d}:00+00:00",
            "action": "BUY" if idx % 3 else "SELL",
            "symbol": "AAPL" if idx % 2 == 0 else "MSFT",
            "model_score": score,
            "threshold": 0.6,
            "quantity": 10 + idx,
            "realized_pnl_total": pnl,
            "unrealized_pnl_total": 0.0,
            "metadata": {
                "intent_score": score,
                "queue_depth": idx % 11,
                "runtime_lane": "swing" if idx % 5 else "futures",
            },
        }
        (rows_a if idx < 600 else rows_b).append(row)

    _write_jsonl(log_dir / "paper_bridge_orders_20260401.jsonl", rows_a)
    _write_jsonl_gz(log_dir / "paper_bridge_orders_20260402.jsonl.gz", rows_b)

    payload = src.build_pytorch_replay_canary(
        project_root,
        max_files=5,
        max_rows=4000,
        validation_fraction=0.2,
        min_rows=200,
        min_class_rows=20,
        epochs=10,
        lr=0.05,
        device="cpu",
        top_fraction=0.1,
    )

    assert payload["ok"] is True
    assert payload["device"] == "cpu"
    assert payload["dataset"]["rows_total"] == 1200
    assert payload["baseline"]["accuracy"] >= 0.9
    assert payload["pytorch"]["accuracy"] >= 0.9
    assert payload["feature_importance"]
    assert payload["sample_quality"] == "full"
    assert "segment_deltas" in payload
    assert "pytorch_calibrated" in payload
    assert "threshold_calibration" in payload
    assert "walk_forward" in payload
    assert payload["walk_forward"]["fold_count"] >= 1
    assert "micro_models" in payload
    assert "mlx_shadow_assist" in payload
    assert payload["threshold_calibration"]["global_threshold"] > 0.0
    assert "keep_mlx_live_default_backend" in payload["recommendations"]


def test_build_pytorch_replay_canary_degrades_instead_of_failing_on_moderate_sample(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    rows = []
    for idx in range(1100):
        score = 0.78 if idx % 2 == 0 else 0.32
        pnl = 1.2 if idx % 2 == 0 else -0.9
        rows.append(
            {
                "timestamp_utc": f"2026-04-03T14:{idx % 60:02d}:00+00:00",
                "action": "BUY" if idx % 3 else "SELL",
                "symbol": "SPY" if idx % 2 == 0 else "QQQ",
                "model_score": score,
                "threshold": 0.6,
                "quantity": 5 + idx,
                "realized_pnl_total": pnl,
                "unrealized_pnl_total": 0.0,
                "metadata": {
                    "intent_score": score,
                    "queue_depth": idx % 7,
                    "runtime_lane": "swing" if idx % 5 else "futures",
                    "layer": "grand_master" if idx % 4 else "sub_bot_paper_mirror",
                    "source_profile": "swing_aggressive" if idx % 5 else "crypto_futures",
                    "shadow_domain": "equities" if idx % 5 else "crypto",
                    "lane_budget_mult": 1.0 if idx % 5 else 0.7,
                    "master_weights": {"trend": 0.3, "mean_revert": 0.2, "shock": 0.5},
                    "allow_live_promotion": True,
                },
            }
        )

    _write_jsonl(log_dir / "paper_bridge_orders_20260403.jsonl", rows)

    payload = src.build_pytorch_replay_canary(
        project_root,
        max_files=5,
        max_rows=4000,
        validation_fraction=0.2,
        min_rows=2000,
        min_class_rows=200,
        epochs=10,
        lr=0.05,
        device="cpu",
        top_fraction=0.1,
    )

    assert payload["ok"] is True
    assert payload["sample_quality"] == "degraded_low_rows_and_class_balance"
    assert payload["warnings"]
    assert payload["effective_dataset_requirements"]["min_rows"] <= 1000
    assert payload["effective_dataset_requirements"]["min_class_rows"] < 200
    assert "threshold_calibration" in payload
    assert "expand_paper_history_for_stronger_pytorch_replay_confidence" in payload["recommendations"]


def test_build_pytorch_replay_canary_calibrates_source_profile_thresholds(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    rows = []
    for idx in range(1400):
        is_aggressive = idx % 2 == 0
        profile = "aggressive" if is_aggressive else "conservative"
        score = 0.85 if is_aggressive else 0.58
        pnl = (1.8 if idx % 6 else -0.8) if is_aggressive else (-0.5 if idx % 4 else 0.7)
        rows.append(
            {
                "timestamp_utc": f"2026-04-05T14:{idx % 60:02d}:00+00:00",
                "action": "BUY" if idx % 3 else "SELL",
                "symbol": "SPY" if is_aggressive else "TLT",
                "model_score": score,
                "threshold": 0.6,
                "quantity": 10 + idx,
                "realized_pnl_total": pnl,
                "unrealized_pnl_total": 0.0,
                "metadata": {
                    "intent_score": score,
                    "queue_depth": idx % 9,
                    "runtime_lane": "day" if is_aggressive else "swing",
                    "source_profile": profile,
                    "shadow_domain": "equities",
                    "layer": "grand_master",
                    "lane_budget_mult": 1.0,
                    "master_weights": {"trend": 0.5, "mean_revert": 0.2, "shock": 0.3},
                    "allow_live_promotion": True,
                },
            }
        )

    _write_jsonl(log_dir / "paper_bridge_orders_20260405.jsonl", rows)

    payload = src.build_pytorch_replay_canary(
        project_root,
        max_files=5,
        max_rows=4000,
        validation_fraction=0.2,
        min_rows=300,
        min_class_rows=20,
        epochs=10,
        lr=0.05,
        device="cpu",
        top_fraction=0.1,
    )

    assert payload["ok"] is True
    calibration = payload["threshold_calibration"]
    assert calibration["segment_count"] >= 2
    assert "aggressive" in calibration["threshold_by_segment"]
    assert "conservative" in calibration["threshold_by_segment"]
    assert payload["pytorch_calibrated"]["selected_count"] > 0
    assert payload["walk_forward"]["fold_count"] >= 1
    assert payload["micro_models"]


def test_history_scoreboard_tracks_recent_runs(tmp_path: Path) -> None:
    history_path = tmp_path / "pytorch_replay_canary_history.jsonl"
    src._append_history(
        history_path,
        {
            "timestamp_utc": "2026-04-01T00:00:00+00:00",
            "ok": True,
            "raw_top_bucket_mean_net_pnl_total_vs_baseline": 1.5,
            "calibrated_selected_mean_net_pnl_total_vs_baseline": 0.5,
            "assist_candidate_count": 1,
        },
    )
    src._append_history(
        history_path,
        {
            "timestamp_utc": "2026-04-02T00:00:00+00:00",
            "ok": False,
            "raw_top_bucket_mean_net_pnl_total_vs_baseline": -2.0,
            "calibrated_selected_mean_net_pnl_total_vs_baseline": -0.25,
            "assist_candidate_count": 0,
        },
    )

    scoreboard = src._history_scoreboard(history_path, limit=10)

    assert scoreboard["runs_tracked"] == 2
    assert scoreboard["ok_runs"] == 1
    assert scoreboard["positive_calibrated_runs"] == 1
    assert scoreboard["active_assist_candidate_runs"] == 1


def test_disabled_payload_keeps_canary_out_of_live_mlx_path(tmp_path: Path) -> None:
    payload = src.disabled_pytorch_replay_canary_payload(tmp_path, tmp_path / "history.jsonl")

    assert payload["ok"] is True
    assert payload["disabled"] is True
    assert payload["mode"] == "disabled_mlx_primary"
    assert "keep_pytorch_replay_canary_disabled_during_live_collection" in payload["recommendations"]
    assert payload["mlx_shadow_assist"]["eligible_source_profiles"] == []
