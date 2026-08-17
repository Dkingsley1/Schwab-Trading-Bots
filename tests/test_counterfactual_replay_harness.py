import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.counterfactual_replay_harness as harness


def test_counterfactual_replay_harness_builds_candidates(tmp_path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_utc": "2026-04-01T14:00:00+00:00",
            "symbol": "AAPL",
            "action": "BUY",
            "strategy": "paper_mirror::alpha",
            "metadata": {"source_profile": "intraday_aggressive"},
            "model_score": 0.71,
            "threshold": 0.60,
            "tradeability_score": 0.50,
            "allocation_conflict_norm": 0.10,
            "realized_pnl_total": 1.0,
            "unrealized_pnl_total": 0.0,
        },
        {
            "timestamp_utc": "2026-04-01T14:05:00+00:00",
            "symbol": "MSFT",
            "action": "BUY",
            "strategy": "paper_mirror::beta",
            "metadata": {"source_profile": "intraday_aggressive"},
            "model_score": 0.58,
            "threshold": 0.60,
            "tradeability_score": 0.20,
            "allocation_conflict_norm": 0.80,
            "realized_pnl_total": -1.0,
            "unrealized_pnl_total": 0.0,
        },
    ]
    (log_dir / "paper_bridge_orders_20260401.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    payload = harness.build_counterfactual_report(project_root, max_rows=100)

    assert payload["ok"] is True
    assert "intraday_aggressive" in payload["profiles_reviewed"]
    assert payload["candidate_count"] > 0
    assert payload["top_candidates"]
    assert payload["processing"]["mode"] == "rebuild"


def test_counterfactual_replay_prefers_event_pnl_over_cumulative_account_totals(tmp_path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_utc": f"2026-04-01T14:0{idx}:00+00:00",
            "symbol": "AAPL",
            "action": "BUY",
            "strategy": "paper_mirror::alpha",
            "metadata": {"source_profile": "intraday_aggressive"},
            "model_score": 0.71,
            "threshold": 0.60,
            "tradeability_score": 0.60,
            "allocation_conflict_norm": 0.10,
            "realized_pnl": 0.25,
            "unrealized_pnl": 0.0,
            "realized_pnl_total": -100.0,
            "unrealized_pnl_total": -25.0,
        }
        for idx in range(4)
    ]
    (log_dir / "paper_bridge_orders_20260401.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    payload = harness.build_counterfactual_report(project_root, max_rows=100)

    assert payload["top_candidates"][0]["aggregate_net_pnl_total"] == 1.0
    assert payload["top_candidates"][0]["win_rate"] == 1.0


def test_counterfactual_replay_prefers_post_cost_delta_over_position_snapshot(tmp_path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": "2026-04-01T14:00:00+00:00",
        "decision_id": "decision-1",
        "symbol": "SH",
        "action": "SELL",
        "strategy": "paper_mirror::short_bias",
        "metadata": {"source_profile": "short_bias_hedge"},
        "model_score": 0.40,
        "threshold": 0.55,
        "tradeability_score": 0.80,
        "allocation_conflict_norm": 0.10,
        "realized_pnl": 1.0,
        "unrealized_pnl": 20.0,
        "post_cost_pnl_delta": -0.25,
    }
    (log_dir / "paper_bridge_orders_20260401.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    payload = harness.build_counterfactual_report(project_root, max_rows=100)

    best = payload["top_candidates"][0]
    assert best["aggregate_net_pnl_total"] == -0.25
    assert best["outcome_source_counts"] == {"post_cost_pnl_delta": 1}
    assert best["attribution_ratio"] == 1.0


def test_counterfactual_replay_uses_action_aware_sell_margin(tmp_path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": "2026-04-01T14:00:00+00:00",
        "decision_id": "sell-1",
        "symbol": "SH",
        "action": "SELL",
        "metadata": {"source_profile": "short_bias_hedge"},
        "model_score": 0.39,
        "threshold": 0.60,
        "tradeability_score": 0.90,
        "allocation_conflict_norm": 0.05,
        "post_cost_pnl_delta": 1.0,
    }
    (log_dir / "paper_bridge_orders_20260401.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    payload = harness.build_counterfactual_report(project_root, max_rows=100)

    assert payload["top_candidates"][0]["aggregate_net_pnl_total"] == 1.0
    assert payload["processing"]["decision_filter_mode"] == "action_aware_margin"


def test_counterfactual_replay_deduplicates_decision_ids(tmp_path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "timestamp_utc": "2026-04-01T14:00:00+00:00",
        "decision_id": "duplicate-decision",
        "symbol": "AAPL",
        "action": "BUY",
        "metadata": {"source_profile": "default"},
        "model_score": 0.70,
        "threshold": 0.60,
        "tradeability_score": 0.80,
        "allocation_conflict_norm": 0.10,
    }
    rows = [{**base, "post_cost_pnl_delta": -5.0}, {**base, "post_cost_pnl_delta": 2.0}]
    (log_dir / "paper_bridge_orders_20260401.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    payload = harness.build_counterfactual_report(project_root, max_rows=100)

    assert payload["processing"]["raw_row_buffer_size"] == 2
    assert payload["processing"]["row_buffer_size"] == 1
    assert payload["processing"]["duplicate_rows_dropped"] == 1
    assert payload["top_candidates"][0]["aggregate_net_pnl_total"] == 2.0


def test_counterfactual_replay_harness_reuses_state_incrementally(tmp_path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_file = project_root / "governance" / "health" / "counterfactual_replay_state.json"
    log_path = log_dir / "paper_bridge_orders_20260401.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-01T14:00:00+00:00",
                "symbol": "AAPL",
                "metadata": {"source_profile": "intraday_aggressive"},
                "model_score": 0.72,
                "threshold": 0.60,
                "tradeability_score": 0.60,
                "allocation_conflict_norm": 0.10,
                "realized_pnl_total": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first = harness.build_counterfactual_report(project_root, max_rows=50, state_file=state_file)
    assert first["processing"]["mode"] == "rebuild"
    assert first["processing"]["full_files"] == 1

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp_utc": "2026-04-01T14:05:00+00:00",
                    "symbol": "MSFT",
                    "metadata": {"source_profile": "intraday_aggressive"},
                    "model_score": 0.74,
                    "threshold": 0.60,
                    "tradeability_score": 0.65,
                    "allocation_conflict_norm": 0.05,
                    "realized_pnl_total": 2.0,
                }
            )
            + "\n"
        )

    second = harness.build_counterfactual_report(project_root, max_rows=50, state_file=state_file)
    assert second["processing"]["mode"] == "incremental"
    assert second["processing"]["incremental_files"] == 1
    assert second["processing"]["row_buffer_size"] == 2


def test_counterfactual_replay_rebuilds_when_consumed_source_is_rewritten_in_place(tmp_path) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_file = project_root / "governance" / "health" / "counterfactual_replay_state.json"
    log_path = log_dir / "paper_bridge_orders_20260401.jsonl"

    first_row = {
        "timestamp_utc": "2026-04-01T14:00:00+00:00",
        "decision_id": "decision-a",
        "symbol": "AAA",
        "action": "BUY",
        "metadata": {"source_profile": "default"},
        "model_score": 0.70,
        "threshold": 0.60,
        "tradeability_score": 0.80,
        "allocation_conflict_norm": 0.10,
        "post_cost_pnl_delta": 1.0,
    }
    second_row = {**first_row, "decision_id": "decision-b", "symbol": "BBB", "post_cost_pnl_delta": 2.0}
    first_text = json.dumps(first_row, sort_keys=True) + "\n"
    second_text = json.dumps(second_row, sort_keys=True) + "\n"
    assert len(first_text) == len(second_text)
    log_path.write_text(first_text, encoding="utf-8")
    first = harness.build_counterfactual_report(project_root, max_rows=50, state_file=state_file)
    assert first["processing"]["mode"] == "rebuild"

    log_path.write_text(second_text, encoding="utf-8")
    second = harness.build_counterfactual_report(project_root, max_rows=50, state_file=state_file)

    assert second["processing"]["mode"] == "rebuild"
    assert second["processing"]["row_buffer_size"] == 1
    assert second["top_candidates"][0]["aggregate_net_pnl_total"] == 2.0
    assert second["processing"]["source_snapshots"][0]["consumed_prefix_fingerprint"]


def test_counterfactual_replay_harness_uses_execution_result_fallback(tmp_path) -> None:
    project_root = tmp_path / "project"
    lane_dir = project_root / "governance" / "execution_lanes"
    lane_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "mode": "paper",
        "timestamp_utc": "2026-04-01T14:00:00+00:00",
        "intent": {
            "target_mode": "paper",
            "timestamp_utc": "2026-04-01T14:00:00+00:00",
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 1,
            "model_score": 0.72,
            "threshold": 0.60,
            "strategy": "paper_mirror::alpha",
            "metadata": {
                "source_profile": "intraday_aggressive",
                "tradeability_score": 0.70,
                "allocation_conflict_norm": 0.10,
            },
        },
        "result": {
            "decision": {
                "timestamp_utc": "2026-04-01T14:00:00+00:00",
                "symbol": "AAPL",
                "action": "BUY",
                "quantity": 1,
                "model_score": 0.72,
                "threshold": 0.60,
                "strategy": "paper_mirror::alpha",
                "realized_pnl": 1.25,
                "unrealized_pnl": 0.0,
                "metadata": {
                    "source_profile": "intraday_aggressive",
                    "tradeability_score": 0.70,
                    "allocation_conflict_norm": 0.10,
                },
            }
        },
    }
    (lane_dir / "execution_results_20260401.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    payload = harness.build_counterfactual_report(project_root, max_rows=100)

    assert payload["ok"] is True
    assert payload["source_files"] == [str(lane_dir / "execution_results_20260401.jsonl")]
    assert "intraday_aggressive" in payload["profiles_reviewed"]
    assert payload["candidate_count"] > 0
    assert payload["top_candidates"][0]["aggregate_net_pnl_total"] > 0
