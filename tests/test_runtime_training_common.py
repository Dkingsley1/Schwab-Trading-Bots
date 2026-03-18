import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import runtime_training_common as rtc


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_load_runtime_observation_sequences_prefers_grand_master_bot(tmp_path) -> None:
    ts = datetime.now(timezone.utc)
    path = tmp_path / "decision_explanations" / "shadow_aggressive_equities" / "decision_explanations_20260313.jsonl"

    _write_jsonl(
        path,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "NVDA",
                "strategy": "grand_master_intent_bot",
                "features": {"last_price": 101.0, "pct_from_close": 0.01},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-1"},
            },
            {
                "timestamp_utc": ts.isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "NVDA",
                "strategy": "grand_master_bot",
                "features": {"last_price": 101.5, "pct_from_close": 0.011},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-1"},
            },
            {
                "timestamp_utc": (ts + timedelta(seconds=90)).isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "NVDA",
                "strategy": "options_master_bot",
                "features": {"last_price": 101.7, "pct_from_close": 0.012},
                "metadata": {"layer": "options_master", "snapshot_id": "snap-2"},
            },
            {
                "timestamp_utc": (ts + timedelta(seconds=180)).isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "NVDA",
                "strategy": "grand_master_bot",
                "features": {"last_price": 102.1, "pct_from_close": 0.014},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-2"},
            },
        ],
    )

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)

    assert ("shadow_aggressive_equities", "NVDA") in sequences
    rows = sequences[("shadow_aggressive_equities", "NVDA")]
    assert len(rows) == 2
    assert rows[0]["strategy"] == "grand_master_bot"
    assert rows[0]["price"] == 101.5
    assert rows[1]["snapshot_id"] == "snap-2"


def test_make_runtime_windowed_dataset_builds_chronological_samples(tmp_path) -> None:
    base_ts = datetime.now(timezone.utc)
    path = tmp_path / "decision_explanations" / "shadow_bond_equities" / "decision_explanations_20260313.jsonl"
    prices = [100.0, 101.0, 102.0, 101.0, 103.0]
    rows = []
    for i, price in enumerate(prices):
        prev_close = prices[max(i - 1, 0)]
        rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(seconds=90 * i)).isoformat(),
                "mode": "shadow_bond_equities",
                "symbol": "TLT",
                "strategy": "grand_master_bot",
                "features": {
                    "last_price": price,
                    "pct_from_close": (price / max(prev_close, 1e-8)) - 1.0,
                    "vol_30m": 0.002 + (0.0001 * i),
                },
                "metadata": {"layer": "grand_master", "snapshot_id": f"snap-{i}"},
            }
        )
    _write_jsonl(path, rows)

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)
    X, y, meta = rtc.make_runtime_windowed_dataset(
        sequences=sequences,
        feature_builder=lambda seq, idx: np.asarray(
            [
                rtc.observation_feature(seq[idx], "pct_from_close"),
                rtc.price_change(seq, idx, 1),
            ],
            dtype=np.float32,
        ),
        label_builder=rtc.direction_label_builder(min_return=0.0),
        window=2,
        horizon=1,
    )

    assert X.shape == (3, 4)
    assert y.shape == (3, 1)
    assert meta["sample_count"] == 3
    assert round(float(meta["positive_rate"]), 4) == 0.6667
    assert list(y[:, 0]) == [1.0, 0.0, 1.0]


def test_risk_support_label_builder_blocks_large_future_drawdown() -> None:
    sequence = [
        {"price": 100.0, "features": {"last_price": 100.0, "vol_30m": 0.001}},
        {"price": 99.9, "features": {"last_price": 99.9, "vol_30m": 0.001}},
        {"price": 96.0, "features": {"last_price": 96.0, "vol_30m": 0.001}},
        {"price": 97.0, "features": {"last_price": 97.0, "vol_30m": 0.001}},
    ]

    label = rtc.risk_support_label_builder(
        min_return=-0.01,
        max_drawdown=0.02,
        max_realized_vol=0.02,
        vol_multiplier=3.0,
    )

    assert label(sequence, 0, 3) == 0.0


def test_selective_direction_label_builder_skips_small_moves() -> None:
    sequence = [
        {"price": 100.0, "features": {"last_price": 100.0}},
        {"price": 100.03, "features": {"last_price": 100.03}},
        {"price": 100.20, "features": {"last_price": 100.20}},
    ]

    label = rtc.selective_direction_label_builder(min_abs_return=0.001)

    assert label(sequence, 0, 1) is None
    assert label(sequence, 0, 2) == 1.0


def test_multi_horizon_direction_label_builder_requires_agreement() -> None:
    aligned = [
        {"price": 100.0, "features": {"last_price": 100.0}},
        {"price": 100.2, "features": {"last_price": 100.2}},
        {"price": 100.5, "features": {"last_price": 100.5}},
    ]
    mixed = [
        {"price": 100.0, "features": {"last_price": 100.0}},
        {"price": 100.2, "features": {"last_price": 100.2}},
        {"price": 99.8, "features": {"last_price": 99.8}},
    ]

    label = rtc.multi_horizon_direction_label_builder(horizons=[1, 2], min_return=0.001)

    assert label(aligned, 0, 2) == 1.0
    assert label(mixed, 0, 2) is None


def test_make_runtime_windowed_dataset_applies_filter_and_confidence_gate(tmp_path) -> None:
    base_ts = datetime.now(timezone.utc)
    path = tmp_path / "decision_explanations" / "shadow_crypto" / "decision_explanations_20260315.jsonl"
    rows = []
    for i, price in enumerate([100.0, 100.4, 100.9, 100.7, 101.1, 101.6]):
        rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(seconds=30 * i)).isoformat(),
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "features": {
                    "last_price": price,
                    "pct_from_close": 0.002 * (i + 1),
                    "quality_gate": 1.0 if i >= 2 else 0.0,
                    "confidence_gate": 0.9 if i >= 3 else 0.2,
                },
                "metadata": {"layer": "grand_master", "snapshot_id": f"snap-{i}"},
            }
        )
    _write_jsonl(path, rows)

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)
    X, y, meta = rtc.make_runtime_windowed_dataset(
        sequences=sequences,
        feature_builder=lambda seq, idx: np.asarray([rtc.observation_feature(seq[idx], "pct_from_close")], dtype=np.float32),
        label_builder=rtc.selective_direction_label_builder(min_abs_return=0.001),
        sample_filter=lambda seq, idx, horizon: rtc.observation_feature(seq[idx], "quality_gate") > 0.5,
        confidence_builder=lambda seq, idx, horizon: rtc.observation_feature(seq[idx], "confidence_gate"),
        min_confidence=0.5,
        window=2,
        horizon=1,
    )

    assert X.shape == (2, 2)
    assert y.shape == (2, 1)
    assert meta["sample_count"] == 2
    assert meta["skipped_filtered"] >= 1
    assert meta["skipped_low_confidence"] >= 1
    assert round(float(meta["confidence_mean"]), 4) == 0.9


def test_load_runtime_observation_sequences_backfills_external_context_for_sparse_rows(tmp_path) -> None:
    ts = datetime.now(timezone.utc)
    path = tmp_path / "decision_explanations" / "shadow_bond_equities" / "decision_explanations_20260318.jsonl"

    _write_jsonl(
        path,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "mode": "shadow_bond_equities",
                "symbol": "TLT",
                "strategy": "grand_master_bot",
                "features": {
                    "last_price": 101.5,
                    "pct_from_close": 0.011,
                    "vol_30m": 0.003,
                },
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-1"},
            }
        ],
    )

    external_root = tmp_path / "data" / "external_context"
    external_root.mkdir(parents=True, exist_ok=True)
    (external_root / "tradingeconomics_latest.json").write_text(
        json.dumps(
            {
                "derived": {
                    "calendar_features": {
                        "calendar_treasury_auction_norm": 0.72,
                        "calendar_macro_surprise_norm": 0.63,
                    },
                    "news_features": {
                        "news_source_quality_norm": 0.84,
                        "news_topic_guidance_norm": 0.41,
                    },
                    "calendar_rows": [],
                }
            }
        ),
        encoding="utf-8",
    )
    (external_root / "market_breadth_latest.json").write_text(
        json.dumps(
            {
                "advancers": 2900,
                "decliners": 1100,
                "up_volume": 4_000_000,
                "down_volume": 1_200_000,
                "new_highs": 180,
                "new_lows": 45,
                "sector_dispersion": 0.018,
            }
        ),
        encoding="utf-8",
    )
    (external_root / "bond_reference_latest.json").write_text(
        json.dumps(
            {
                "treasury_yields": {"2y": 4.1, "5y": 4.0, "10y": 4.2, "30y": 4.4, "real_10y": 1.8},
                "symbols": {
                    "TLT": {
                        "duration_years_norm": 0.72,
                        "nav_discount_norm": 0.49,
                        "flow_5d_norm": 0.18,
                        "ytm_norm": 0.51,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (external_root / "live_macro_latest.json").write_text(
        json.dumps(
            {
                "active": True,
                "template": "powell",
                "source": "Federal Reserve",
                "broad_market": True,
                "sentiment_hint": -0.75,
                "shock_hint": 1.0,
            }
        ),
        encoding="utf-8",
    )

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)
    row = sequences[("shadow_bond_equities", "TLT")][0]
    features = row["features"]

    assert features["calendar_treasury_auction_norm"] == 0.72
    assert features["breadth_risk_off_norm"] > 0.0
    assert features["bond_yield_10y_norm"] > 0.0
    assert features["news_available"] > 0.0
    assert features["news_sentiment"] < 0.0


def test_load_runtime_observation_sequences_carries_forward_recent_context(tmp_path) -> None:
    base_ts = datetime.now(timezone.utc)
    path = tmp_path / "decision_explanations" / "shadow_bond_equities" / "decision_explanations_20260318.jsonl"

    _write_jsonl(
        path,
        [
            {
                "timestamp_utc": base_ts.isoformat(),
                "mode": "shadow_bond_equities",
                "symbol": "TLT",
                "strategy": "grand_master_bot",
                "features": {
                    "last_price": 100.0,
                    "pct_from_close": 0.010,
                    "bond_curve_2s10s_norm": 0.77,
                },
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-1"},
            },
            {
                "timestamp_utc": (base_ts + timedelta(seconds=90)).isoformat(),
                "mode": "shadow_bond_equities",
                "symbol": "TLT",
                "strategy": "grand_master_bot",
                "features": {
                    "last_price": 100.4,
                    "pct_from_close": 0.004,
                },
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-2"},
            },
        ],
    )

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)
    rows = sequences[("shadow_bond_equities", "TLT")]

    assert rows[0]["features"]["bond_curve_2s10s_norm"] == 0.77
    assert rows[1]["features"]["bond_curve_2s10s_norm"] == 0.77
