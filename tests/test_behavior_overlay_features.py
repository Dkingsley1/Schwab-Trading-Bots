from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import scripts.build_behavior_dataset_from_decisions as behavior_ds
import scripts.run_shadow_training_loop as loop
import scripts.train_trade_behavior_bot as trainer


def test_behavior_feature_schema_appends_lane_overlay_features() -> None:
    assert behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES == loop._BEHAVIOR_LANE_FEATURE_NAMES
    capital_tail = behavior_ds.BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES
    lane_tail_end = len(behavior_ds.FEATURE_NAMES) - len(capital_tail)
    lane_tail_start = lane_tail_end - len(behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES)
    assert behavior_ds.FEATURE_NAMES[lane_tail_start:lane_tail_end] == behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES
    assert loop._BEHAVIOR_FEATURE_NAMES_V2[lane_tail_start:lane_tail_end] == loop._BEHAVIOR_LANE_FEATURE_NAMES
    assert behavior_ds.FEATURE_NAMES[-len(capital_tail) :] == capital_tail
    assert loop._BEHAVIOR_FEATURE_NAMES_V2[-len(loop._BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES) :] == loop._BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES
    assert behavior_ds.BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES == loop._BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES


def test_load_governance_index_captures_lane_strategy_features() -> None:
    since_utc = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = [
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_id": "snap-1",
            "lane_strategy_features": {
                "day_regime_trend_norm": 0.81,
                "swing_regime_chop_norm": 0.24,
                "bond_curve_steepener_norm": 0.67,
            },
        }
    ]

    out = behavior_ds._load_governance_index(rows, since_utc=since_utc)

    assert out["snap-1"]["day_regime_trend_norm"] == 0.81
    assert out["snap-1"]["swing_regime_chop_norm"] == 0.24
    assert out["snap-1"]["bond_curve_steepener_norm"] == 0.67
    assert out["snap-1"]["day_execution_cost_risk_norm"] == 0.0


def test_load_governance_index_captures_capital_flow_features() -> None:
    since_utc = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = [
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_id": "snap-2",
            "capital_flow": {
                "capital_flow_signed_scaled": -0.72,
                "capital_flow_inflow_norm": 0.0,
                "capital_flow_outflow_norm": 0.61,
            },
        }
    ]

    out = behavior_ds._load_governance_index(rows, since_utc=since_utc)

    assert out["snap-2"]["capital_flow_signed_scaled"] == -0.72
    assert out["snap-2"]["capital_flow_inflow_norm"] == 0.0
    assert out["snap-2"]["capital_flow_outflow_norm"] == 0.61


def test_behavior_feature_vector_v2_accepts_appended_lane_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "NVDA",
        "BUY",
        {
            "pct_from_close": 0.01,
            "mom_5m": 0.004,
            "vol_30m": 0.008,
            "range_pos": 0.8,
            "spread_bps": 3.0,
            "day_regime_trend_norm": 0.83,
            "swing_regime_alignment_norm": 0.66,
            "bond_carry_roll_norm": 0.41,
        },
        {},
    )

    assert vec is not None
    assert vec.shape[1] == len(loop._BEHAVIOR_FEATURE_NAMES_V2)
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("day_regime_trend_norm")] == 0.83
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("swing_regime_alignment_norm")] == 0.66
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("bond_carry_roll_norm")] == 0.41


def test_behavior_feature_vector_v2_accepts_capital_flow_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "SPY",
        "SELL",
        {
            "pct_from_close": -0.004,
            "mom_5m": -0.002,
            "vol_30m": 0.005,
            "capital_flow_signed_scaled": -0.88,
            "capital_flow_inflow_norm": 0.0,
            "capital_flow_outflow_norm": 0.74,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("capital_flow_signed_scaled")] == -0.88
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("capital_flow_inflow_norm")] == 0.0
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("capital_flow_outflow_norm")] == 0.74


def test_effective_account_equity_proxy_prefers_fresh_broker_truth() -> None:
    effective, meta = loop._effective_account_equity_proxy(
        {
            "status": "mismatch",
            "age_iters": 1,
            "account_metrics": {"equity": 152500.0, "cash_balance": 48000.0},
        },
        fallback_equity_proxy=100000.0,
    )

    assert effective == 152500.0
    assert meta["source"] == "broker_truth_account_metrics"


def test_estimate_capital_flow_state_detects_large_outflow() -> None:
    flow = loop._estimate_capital_flow_state(
        {"equity": 87000.0, "cash_balance": 22000.0},
        {"equity": 120000.0, "cash_balance": 55000.0},
    )

    assert flow["detected"] is True
    assert flow["estimated_amount"] < 0.0
    assert flow["capital_flow_signed_scaled"] < 0.0
    assert flow["capital_flow_outflow_norm"] > 0.0
    assert flow["capital_flow_inflow_norm"] == 0.0


def test_rollback_schema_compatible_allows_prefix_compatible_extension() -> None:
    ok, reason = trainer._rollback_schema_compatible(
        {
            "load_ok": True,
            "effective_dim": 3,
            "feature_names": ["a", "b", "c"],
        },
        dataset_feature_dim=5,
        dataset_feature_names=["a", "b", "c", "d", "e"],
        require_feature_names=True,
    )

    assert ok is True
    assert reason.startswith("prefix_compatible")


def test_curated_dataset_guard_accepts_curated_behavior_dataset() -> None:
    ok, reason, summary = trainer._curated_dataset_guard(
        {
            "dataset_kind": "curated_decision_governance",
            "source": {
                "decision_files": 3,
                "decision_sql_files": 2,
                "governance_files": 2,
                "governance_sql_files": 1,
                "pnl_attribution_files": 1,
                "pnl_sql_files": 1,
            },
        }
    )

    assert ok is True
    assert reason == "ok"
    assert summary["decision_sources"] == 5
    assert summary["governance_sources"] == 3


def test_curated_dataset_guard_rejects_legacy_dataset_kind() -> None:
    ok, reason, summary = trainer._curated_dataset_guard(
        {
            "dataset_kind": "legacy_trade_history",
            "source": {
                "decision_files": 3,
                "governance_files": 2,
            },
        }
    )

    assert ok is False
    assert reason == "dataset_kind_not_curated"
    assert summary["dataset_kind"] == "legacy_trade_history"
