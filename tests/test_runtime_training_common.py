import json
import sqlite3
import sys
import gzip
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import runtime_training_common as rtc


def test_safe_int_handles_numeric_strings_and_bad_values() -> None:
    assert rtc._safe_int("12.9", 5) == 12
    assert rtc._safe_int("bad", 5) == 5
    assert rtc._safe_int(float("nan"), 5) == 5


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row) for row in rows) + "\n")


def _write_runtime_sqlite(path: Path, source_rel: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            run_id TEXT,
            iter_id TEXT,
            decision_id TEXT,
            parent_decision_id TEXT,
            log_schema_version INTEGER,
            UNIQUE(source_file, line_no)
        )
        """
    )
    for idx, row in enumerate(rows, start=1):
        conn.execute(
            """
            INSERT INTO jsonl_records (
                source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json,
                run_id, iter_id, decision_id, parent_decision_id, log_schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(path),
                source_rel,
                idx,
                datetime.now(timezone.utc).isoformat(),
                f"sha-{idx}",
                json.dumps(row),
                "",
                "",
                "",
                "",
                2,
            ),
        )
    conn.commit()
    conn.close()


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


def test_load_runtime_observation_sequences_honors_max_observation_rows(tmp_path) -> None:
    ts = datetime.now(timezone.utc)
    path = tmp_path / "decision_explanations" / "shadow_aggressive_equities" / "decision_explanations_20260313.jsonl"

    _write_jsonl(
        path,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "NVDA",
                "strategy": "grand_master_bot",
                "features": {"last_price": 101.0},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-1"},
            },
            {
                "timestamp_utc": (ts + timedelta(seconds=90)).isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "AAPL",
                "strategy": "grand_master_bot",
                "features": {"last_price": 202.0},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-2"},
            },
        ],
    )

    sequences = rtc.load_runtime_observation_sequences(
        tmp_path,
        lookback_days=2,
        allow_snapshot=False,
        max_observation_rows=1,
    )

    assert ("shadow_aggressive_equities", "NVDA") in sequences
    assert ("shadow_aggressive_equities", "AAPL") not in sequences


def test_load_runtime_observation_sequences_can_use_sqlite_first(tmp_path, monkeypatch) -> None:
    ts = datetime.now(timezone.utc)
    rel = "decision_explanations/shadow_aggressive_equities/decision_explanations_20260313.jsonl"
    file_path = tmp_path / rel
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("", encoding="utf-8")
    sqlite_path = tmp_path / "data" / "jsonl_link.sqlite3"
    _write_runtime_sqlite(
        sqlite_path,
        rel,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "NVDA",
                "strategy": "grand_master_bot",
                "features": {"last_price": 101.5, "pct_from_close": 0.011},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-sql"},
            }
        ],
    )

    monkeypatch.setenv("RUNTIME_TRAIN_SQLITE_PATH", str(sqlite_path))
    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2, prefer_sqlite=True, allow_snapshot=False)

    rows = sequences[("shadow_aggressive_equities", "NVDA")]
    assert len(rows) == 1
    assert rows[0]["snapshot_id"] == "snap-sql"


def test_load_runtime_observation_sequences_can_use_sqlite_history_without_raw_files(tmp_path, monkeypatch) -> None:
    ts = datetime.now(timezone.utc)
    day = ts.strftime("%Y%m%d")
    rel = f"decision_explanations/shadow_aggressive_equities/decision_explanations_{day}.jsonl.gz"
    sqlite_path = tmp_path / "data" / "jsonl_link.sqlite3"
    _write_runtime_sqlite(
        sqlite_path,
        rel,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "NVDA",
                "strategy": "grand_master_bot",
                "features": {"last_price": 101.5, "pct_from_close": 0.011},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-sql-only"},
            }
        ],
    )

    monkeypatch.setenv("RUNTIME_TRAIN_SQLITE_PATH", str(sqlite_path))
    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2, prefer_sqlite=True, allow_snapshot=False)

    rows = sequences[("shadow_aggressive_equities", "NVDA")]
    assert len(rows) == 1
    assert rows[0]["snapshot_id"] == "snap-sql-only"


def test_load_runtime_observation_sequences_accepts_trade_decision_metadata_mode(tmp_path) -> None:
    ts = datetime.now(timezone.utc)
    day = ts.strftime("%Y%m%d")
    path = tmp_path / "decisions" / "paper" / f"trade_decisions_{day}.jsonl"
    _write_jsonl(
        path,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "symbol": "SOXX",
                "strategy": "paper_mirror::brain_refinery_v35_dmi_state_machine",
                "features": {"last_price": 536.51, "pct_from_close": 0.011},
                "metadata": {
                    "mode": "paper",
                    "layer": "sub_bot_paper_mirror",
                    "snapshot_id": "snap-paper-trade",
                },
            }
        ],
    )

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2, allow_snapshot=False)

    rows = sequences[("paper", "SOXX")]
    assert len(rows) == 1
    assert rows[0]["snapshot_id"] == "snap-paper-trade"
    assert rows[0]["price"] == 536.51


def test_load_runtime_observation_sequences_infers_crypto_channel_modes(tmp_path) -> None:
    ts = datetime.now(timezone.utc)
    day = ts.strftime("%Y%m%d")
    default_path = tmp_path / "decisions" / "shadow_crypto" / f"trade_decisions_{day}.jsonl"
    futures_path = tmp_path / "decisions" / "shadow_crypto_futures_crypto" / f"trade_decisions_{day}.jsonl"
    _write_jsonl(
        default_path,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "snapshot_id": "btc-default-snap",
                "features": {"last_price": 62000.0, "pct_from_close": 0.001, "vol_30m": 0.002},
                "metadata": {
                    "mode": "shadow",
                    "layer": "grand_master",
                    "source_profile": "default",
                    "shadow_domain": "crypto",
                    "snapshot_id": "btc-default-snap",
                },
            }
        ],
    )
    _write_jsonl(
        futures_path,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "symbol": "ETH-USD",
                "strategy": "grand_master_bot",
                "snapshot_id": "eth-futures-snap",
                "features": {"last_price": 1600.0, "pct_from_close": -0.002, "vol_30m": 0.003},
                "metadata": {
                    "mode": "shadow",
                    "layer": "grand_master",
                    "source_profile": "crypto_futures",
                    "shadow_domain": "crypto",
                    "snapshot_id": "eth-futures-snap",
                },
            }
        ],
    )

    sequences = rtc.load_runtime_observation_sequences(
        tmp_path,
        lookback_days=2,
        mode_allowlist=["shadow_crypto", "shadow_crypto_futures_crypto"],
        prefer_sqlite=False,
        allow_snapshot=False,
    )

    assert ("shadow_crypto", "BTC-USD") in sequences
    assert ("shadow_crypto_futures_crypto", "ETH-USD") in sequences
    assert sequences[("shadow_crypto", "BTC-USD")][0]["strategy"] == "grand_master_bot"
    assert sequences[("shadow_crypto", "BTC-USD")][0]["price"] == 62000.0
    assert sequences[("shadow_crypto_futures_crypto", "ETH-USD")][0]["features"]["pct_from_close"] == -0.002


def test_load_runtime_observation_sequences_recovers_split_channel_price(tmp_path) -> None:
    ts = datetime.now(timezone.utc)
    day = ts.strftime("%Y%m%d")
    channel_path = (
        tmp_path
        / "governance"
        / "channels"
        / "decision"
        / "schwab_futures_equities_schwab"
        / f"decision_{day}.jsonl"
    )
    trade_path = tmp_path / "decisions" / "shadow_schwab_futures_equities" / f"trade_decisions_{day}.jsonl"
    _write_jsonl(
        channel_path,
        [
            {
                "timestamp_utc": (ts - timedelta(seconds=30)).isoformat(),
                "symbol": "SPY",
                "snapshot_id": "SPY-split-snapshot",
                "market": {"last_price": 753.13},
            }
        ],
    )
    _write_jsonl(
        trade_path,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "symbol": "SPY",
                "strategy": "master_trend_bot",
                "features": {"pct_from_close": 0.012},
                "gates": {"market_data_ok": True},
                "metadata": {
                    "mode": "shadow",
                    "layer": "master_bot",
                    "snapshot_id": "SPY-split-snapshot",
                },
            }
        ],
    )

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2, prefer_sqlite=False, allow_snapshot=False)

    rows = sequences[("shadow", "SPY")]
    assert len(rows) == 1
    assert rows[0]["snapshot_id"] == "SPY-split-snapshot"
    assert rows[0]["price"] == 753.13
    assert rows[0]["features"]["last_price"] == 753.13
    assert rows[0]["features"]["price_recovered_from_sidecar"] == 1.0


def test_load_runtime_observation_sequences_accepts_sqlite_trade_history_without_raw_files(tmp_path, monkeypatch) -> None:
    ts = datetime.now(timezone.utc)
    day = ts.strftime("%Y%m%d")
    rel = f"decisions/paper/trade_decisions_{day}.jsonl.gz"
    sqlite_path = tmp_path / "data" / "jsonl_link.sqlite3"
    _write_runtime_sqlite(
        sqlite_path,
        rel,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "symbol": "SOXX",
                "strategy": "paper_mirror::brain_refinery_v35_dmi_state_machine",
                "features": {"last_price": 536.51, "pct_from_close": 0.011},
                "metadata": {
                    "mode": "paper",
                    "layer": "sub_bot_paper_mirror",
                    "snapshot_id": "snap-sql-trade",
                },
            }
        ],
    )

    monkeypatch.setenv("RUNTIME_TRAIN_SQLITE_PATH", str(sqlite_path))
    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2, prefer_sqlite=True, allow_snapshot=False)

    rows = sequences[("paper", "SOXX")]
    assert len(rows) == 1
    assert rows[0]["snapshot_id"] == "snap-sql-trade"


def test_load_runtime_observation_sequences_falls_back_to_raw_when_sql_merge_is_busy(tmp_path, monkeypatch) -> None:
    ts = datetime.now(timezone.utc)
    rel = "decision_explanations/shadow_aggressive_equities/decision_explanations_20260313.jsonl"
    file_path = tmp_path / rel
    _write_jsonl(
        file_path,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "NVDA",
                "strategy": "grand_master_bot",
                "features": {"last_price": 101.25, "pct_from_close": 0.0105},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-raw"},
            }
        ],
    )

    sqlite_path = tmp_path / "data" / "jsonl_link.sqlite3"
    _write_runtime_sqlite(
        sqlite_path,
        rel,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "NVDA",
                "strategy": "grand_master_bot",
                "features": {"last_price": 101.75, "pct_from_close": 0.0115},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-sql"},
            }
        ],
    )

    progress_path = tmp_path / "governance" / "health" / "sql_link_service_progress_latest.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "current_step": "merge_primary",
                "running": True,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("RUNTIME_TRAIN_SQLITE_PATH", str(sqlite_path))
    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2, prefer_sqlite=True, allow_snapshot=False)

    rows = sequences[("shadow_aggressive_equities", "NVDA")]
    assert len(rows) == 1
    assert rows[0]["snapshot_id"] == "snap-raw"


def test_load_runtime_observation_sequences_can_use_snapshot_file(tmp_path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    rows_path = tmp_path / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_row = {
        "mode": "shadow_bond_equities",
        "symbol": "TLT",
        "strategy": "grand_master_bot",
        "strategy_priority": 0,
        "snapshot_id": "snap-cache",
        "ts_epoch": float(now.timestamp()),
        "timestamp_utc": now.isoformat(),
        "price": 101.0,
        "features": {"last_price": 101.0, "pct_from_close": 0.01},
    }
    rows_path.write_text(json.dumps(snapshot_row) + "\n", encoding="utf-8")
    health_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "lookback_days": 14,
                "rows_path": str(rows_path),
                "row_count": 1,
                "sequence_count": 1,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("RUNTIME_TRAIN_USE_SNAPSHOT", "1")
    monkeypatch.setenv("RUNTIME_TRAIN_SNAPSHOT_FILE", str(health_path))
    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)

    assert ("shadow_bond_equities", "TLT") in sequences
    assert sequences[("shadow_bond_equities", "TLT")][0]["snapshot_id"] == "snap-cache"


def test_load_runtime_observation_sequences_snapshot_only_skips_history_fallback(tmp_path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    rows_path = tmp_path / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        json.dumps(
            {
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "strategy_priority": 0,
                "snapshot_id": "snap-nonmatching",
                "ts_epoch": float(now.timestamp()),
                "timestamp_utc": now.isoformat(),
                "price": 100000.0,
                "features": {"last_price": 100000.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    health_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "lookback_days": 14,
                "rows_path": str(rows_path),
                "row_count": 1,
                "sequence_count": 1,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        tmp_path / "decision_explanations" / "shadow_bond_equities" / f"decision_explanations_{now:%Y%m%d}.jsonl",
        [
            {
                "timestamp_utc": now.isoformat(),
                "mode": "shadow_bond_equities",
                "symbol": "TLT",
                "strategy": "grand_master_bot",
                "features": {"last_price": 101.5},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-history"},
            }
        ],
    )

    monkeypatch.setenv("RUNTIME_TRAIN_USE_SNAPSHOT", "1")
    monkeypatch.setenv("RUNTIME_TRAIN_SNAPSHOT_ONLY", "1")
    monkeypatch.setenv("RUNTIME_TRAIN_SNAPSHOT_FILE", str(health_path))
    sequences = rtc.load_runtime_observation_sequences(
        tmp_path,
        lookback_days=2,
        mode_allowlist=["shadow_bond_equities"],
        symbol_allowlist=["TLT"],
    )

    assert sequences == {}


def test_load_runtime_observation_sequences_reads_gzip_history(tmp_path) -> None:
    ts = datetime.now(timezone.utc)
    path = tmp_path / "decision_explanations" / "shadow_conservative_equities" / "decision_explanations_20260313.jsonl.gz"
    _write_jsonl_gz(
        path,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "mode": "shadow_conservative_equities",
                "symbol": "TLT",
                "strategy": "grand_master_bot",
                "features": {"last_price": 101.5, "pct_from_close": 0.011},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-gz"},
            }
        ],
    )

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=30, allow_snapshot=False)

    rows = sequences[("shadow_conservative_equities", "TLT")]
    assert len(rows) == 1
    assert rows[0]["snapshot_id"] == "snap-gz"


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


def test_runtime_label_evidence_is_deterministic_and_point_in_time_safe() -> None:
    base_ts = datetime.now(timezone.utc)
    sequence = [
        {
            "timestamp_utc": (base_ts + timedelta(minutes=i)).isoformat(),
            "mode": "paper",
            "symbol": "SPY",
            "snapshot_id": f"snap-{i}",
            "price": 100.0 + i,
            "features": {"last_price": 100.0 + i},
        }
        for i in range(3)
    ]

    first = rtc.runtime_label_evidence(
        sequence,
        0,
        2,
        expected_mode="paper",
        expected_symbol="SPY",
        label_owner_id="brain_refinery_v1",
    )
    second = rtc.runtime_label_evidence(
        sequence,
        0,
        2,
        expected_mode="paper",
        expected_symbol="SPY",
        label_owner_id="brain_refinery_v1",
    )

    assert first["eligible"] is True
    assert first["maturity_seconds"] == 120.0
    assert first["lineage_sha256"] == second["lineage_sha256"]
    assert len(first["lineage_sha256"]) == 64


def test_runtime_label_evidence_rejects_cross_symbol_and_noncausal_outcomes() -> None:
    base_ts = datetime.now(timezone.utc)
    sequence = [
        {
            "timestamp_utc": base_ts.isoformat(),
            "mode": "paper",
            "symbol": "SPY",
            "snapshot_id": "same-snapshot",
            "price": 100.0,
        },
        {
            "timestamp_utc": (base_ts - timedelta(seconds=1)).isoformat(),
            "mode": "paper",
            "symbol": "QQQ",
            "snapshot_id": "same-snapshot",
            "price": 101.0,
        },
    ]

    evidence = rtc.runtime_label_evidence(
        sequence,
        0,
        1,
        expected_mode="paper",
        expected_symbol="SPY",
    )

    assert evidence["eligible"] is False
    assert "cross_symbol_label_join" in evidence["reasons"]
    assert "noncausal_label_maturity" in evidence["reasons"]
    assert "duplicate_snapshot_label_join" in evidence["reasons"]

    missing_timestamps = rtc.runtime_label_evidence(
        [
            {"mode": "paper", "symbol": "SPY", "snapshot_id": "a", "price": 100.0},
            {"mode": "paper", "symbol": "SPY", "snapshot_id": "b", "price": 101.0},
        ],
        0,
        1,
        expected_mode="paper",
        expected_symbol="SPY",
    )
    assert "missing_feature_timestamp" in missing_timestamps["reasons"]
    assert "missing_label_maturity_timestamp" in missing_timestamps["reasons"]


def test_runtime_dataset_refuses_market_proxy_labels_for_operational_objectives() -> None:
    base_ts = datetime.now(timezone.utc)
    rows = [
        {
            "timestamp_utc": (base_ts + timedelta(minutes=i)).isoformat(),
            "mode": "paper",
            "symbol": "SPY",
            "snapshot_id": f"op-{i}",
            "ts_epoch": float((base_ts + timedelta(minutes=i)).timestamp()),
            "price": 100.0 + i,
            "features": {"last_price": 100.0 + i},
        }
        for i in range(5)
    ]

    X, y, meta = rtc.make_runtime_windowed_dataset(
        sequences={("paper", "SPY"): rows},
        feature_builder=lambda seq, idx: np.asarray([rtc.observation_feature(seq[idx], "last_price")]),
        label_builder=rtc.direction_label_builder(),
        label_contract={"objective_class": "operational_effect"},
        label_owner_id="brain_refinery_v1000_infrastructure_bot",
        window=2,
        horizon=1,
    )

    assert X.shape == (0, 0)
    assert y.shape == (0, 1)
    audit = meta["label_evidence_audit"]
    assert audit["training_eligible"] is False
    assert audit["rejected_evidence_candidate_count"] == 3
    assert audit["rejection_reason_occurrence_count"] == 3
    assert audit["rejection_counts"]["objective_requires_non_market_outcome"] == 3


def test_runtime_dataset_counts_rejected_candidates_separately_from_reasons() -> None:
    base_ts = datetime.now(timezone.utc)
    rows = [
        {
            "timestamp_utc": base_ts.isoformat(),
            "mode": "paper",
            "symbol": "SPY",
            "snapshot_id": "duplicate",
            "price": 100.0,
        },
        {
            "timestamp_utc": (base_ts - timedelta(seconds=1)).isoformat(),
            "mode": "paper",
            "symbol": "QQQ",
            "snapshot_id": "duplicate",
            "price": 101.0,
        },
    ]

    _, _, meta = rtc.make_runtime_windowed_dataset(
        sequences={("paper", "SPY"): rows},
        feature_builder=lambda seq, idx: np.asarray([1.0], dtype=np.float32),
        label_builder=rtc.direction_label_builder(),
        label_contract={"objective_class": "market_outcome"},
        label_owner_id="brain_refinery_v1",
        window=1,
        horizon=1,
    )

    audit = meta["label_evidence_audit"]
    assert audit["candidate_count"] == 1
    assert audit["rejected_evidence_candidate_count"] == 1
    assert audit["rejection_reason_occurrence_count"] == 3
    assert audit["selected_training_sample_count"] == 0


def test_runtime_dataset_selects_outcome_at_semantic_wall_clock_horizon() -> None:
    base_ts = datetime.now(timezone.utc)
    offsets = [0, 10 * 60, 20 * 60, 24 * 60 * 60, 25 * 60 * 60]
    rows = [
        {
            "timestamp_utc": (base_ts + timedelta(seconds=offset)).isoformat(),
            "mode": "paper",
            "symbol": "SPY",
            "snapshot_id": f"daily-{index}",
            "price": 100.0 + index,
            "features": {"last_price": 100.0 + index},
        }
        for index, offset in enumerate(offsets)
    ]
    contract = {
        "objective_class": "market_outcome",
        "primary_horizon": "1d_forward_return",
        "minimum_label_maturity_seconds": 24 * 60 * 60,
        "maximum_label_maturity_seconds": 4 * 24 * 60 * 60,
        "label_horizon_policy": {"enforcement_mode": "strict_wall_clock_range"},
    }

    X, _, meta = rtc.make_runtime_windowed_dataset(
        sequences={("paper", "SPY"): rows},
        feature_builder=lambda seq, idx: np.asarray([rtc.observation_feature(seq[idx], "last_price")]),
        label_builder=rtc.direction_label_builder(),
        label_contract=contract,
        label_owner_id="brain_refinery_v10_seasonal",
        window=1,
        horizon=1,
    )

    audit = meta["label_evidence_audit"]
    assert X.shape[0] == 3
    assert audit["candidate_count"] == 4
    assert audit["selected_training_sample_count"] == 3
    assert audit["rejection_counts"]["label_horizon_not_mature_for_contract"] == 1
    assert audit["maturity_seconds_min"] >= 24 * 60 * 60
    assert audit["effective_horizon_rows_min"] == 2
    assert audit["effective_horizon_rows_max"] == 3


def test_runtime_dataset_rejects_outcome_after_contract_maximum() -> None:
    base_ts = datetime.now(timezone.utc)
    rows = [
        {
            "timestamp_utc": base_ts.isoformat(),
            "mode": "paper",
            "symbol": "SPY",
            "snapshot_id": "late-0",
            "price": 100.0,
        },
        {
            "timestamp_utc": (base_ts + timedelta(days=5)).isoformat(),
            "mode": "paper",
            "symbol": "SPY",
            "snapshot_id": "late-1",
            "price": 101.0,
        },
    ]

    X, _, meta = rtc.make_runtime_windowed_dataset(
        sequences={("paper", "SPY"): rows},
        feature_builder=lambda seq, idx: np.asarray([1.0], dtype=np.float32),
        label_builder=rtc.direction_label_builder(),
        label_contract={
            "objective_class": "market_outcome",
            "primary_horizon": "1d_forward_return",
            "minimum_label_maturity_seconds": 24 * 60 * 60,
            "maximum_label_maturity_seconds": 4 * 24 * 60 * 60,
        },
        window=1,
        horizon=1,
    )

    assert X.shape == (0, 0)
    assert meta["label_evidence_audit"]["rejection_counts"]["label_maturity_after_contract_maximum"] == 1


def test_make_runtime_windowed_dataset_label_repair_can_recover_abstained_samples(tmp_path) -> None:
    base_ts = datetime.now(timezone.utc)
    prices = [100.0, 100.4, 100.1, 100.8, 100.2, 101.0]
    rows = []
    for i, price in enumerate(prices):
        rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(seconds=90 * i)).isoformat(),
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "strategy_priority": 0,
                "snapshot_id": f"snap-repair-{i}",
                "ts_epoch": float((base_ts + timedelta(seconds=90 * i)).timestamp()),
                "price": price,
                "features": {"last_price": price, "pct_from_close": (price / prices[max(i - 1, 0)]) - 1.0},
            }
        )
    sequences = {("shadow_crypto", "BTC-USD"): rows}

    X, y, meta = rtc.make_runtime_windowed_dataset(
        sequences=sequences,
        feature_builder=lambda seq, idx: np.asarray([rtc.observation_feature(seq[idx], "pct_from_close")], dtype=np.float32),
        label_builder=lambda seq, idx, horizon: None,
        sample_filter=lambda seq, idx, horizon: False,
        bypass_sample_filter=True,
        fallback_direction_label=True,
        fallback_min_abs_return=0.0001,
        window=2,
        horizon=1,
    )

    assert X.shape[0] == 4
    assert y.shape == (4, 1)
    assert meta["label_repair_enabled"] is True
    assert meta["label_repair_bypassed_filter"] is True
    assert meta["label_repaired"] == 4
    assert meta["skipped_filtered"] == 0


def test_runtime_feature_registries_include_new_trading_context_keys() -> None:
    for key in (
        "market_micro_lunch_chop_norm",
        "market_micro_gap_fade_risk_norm",
        "options_iv_crush_risk_norm",
        "options_spread_execution_risk_norm",
        "options_higher_order_greek_pressure_norm",
        "options_volatility_arbitrage_proxy_norm",
        "fx_session_london_norm",
        "fx_dxy_yield_confirmation_norm",
        "schwab_education_item_density_norm",
        "schwab_education_symbol_frequency_norm",
    ):
        assert key in rtc._RUNTIME_GAP_FILL_KEYS


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


def test_cost_adjusted_direction_label_builder_filters_small_edges() -> None:
    sequence = [
        {
            "price": 100.0,
            "features": {
                "last_price": 100.0,
                "spread_bps": 28.0,
                "market_micro_tradeability_score_norm": 0.28,
                "futures_vwap_bias_norm": 0.72,
            },
        },
        {
            "price": 100.18,
            "features": {
                "last_price": 100.18,
                "spread_bps": 6.0,
                "market_micro_tradeability_score_norm": 0.92,
                "futures_vwap_bias_norm": 0.52,
            },
        },
        {
            "price": 101.10,
            "features": {
                "last_price": 101.10,
                "spread_bps": 6.0,
                "market_micro_tradeability_score_norm": 0.92,
                "futures_vwap_bias_norm": 0.52,
            },
        },
    ]

    label = rtc.cost_adjusted_direction_label_builder(min_edge=0.0015, transaction_cost_bps=8.0)

    assert label(sequence, 0, 1) is None
    assert label(sequence, 0, 2) == 1.0


def test_fill_adjusted_outcome_label_builder_requires_post_cost_edge() -> None:
    sequence = [
        {
            "price": 100.0,
            "features": {
                "last_price": 100.0,
                "spread_bps": 22.0,
                "lag_slippage_bps": 8.0,
                "lag_impact_bps": 5.0,
                "lag_fee_bps": 2.0,
                "execution_fitness_norm": 0.34,
                "stop_target_realism_norm": 0.42,
            },
        },
        {
            "price": 100.20,
            "features": {
                "last_price": 100.20,
                "spread_bps": 8.0,
                "lag_slippage_bps": 2.0,
                "lag_impact_bps": 1.0,
                "lag_fee_bps": 1.0,
                "execution_fitness_norm": 0.86,
                "stop_target_realism_norm": 0.78,
            },
        },
        {
            "price": 101.30,
            "features": {
                "last_price": 101.30,
                "spread_bps": 8.0,
                "lag_slippage_bps": 2.0,
                "lag_impact_bps": 1.0,
                "lag_fee_bps": 1.0,
                "execution_fitness_norm": 0.86,
                "stop_target_realism_norm": 0.78,
            },
        },
    ]

    label = rtc.fill_adjusted_outcome_label_builder(min_net_return=0.0015, transaction_cost_bps=8.0)

    assert label(sequence, 0, 1) is None
    assert label(sequence, 0, 2) == 1.0


def test_event_followthrough_label_builder_requires_same_direction_path() -> None:
    aligned = [
        {"price": 100.0, "features": {"last_price": 100.0}},
        {"price": 100.5, "features": {"last_price": 100.5}},
        {"price": 100.8, "features": {"last_price": 100.8}},
        {"price": 101.2, "features": {"last_price": 101.2}},
    ]
    whipsaw = [
        {"price": 100.0, "features": {"last_price": 100.0}},
        {"price": 99.2, "features": {"last_price": 99.2}},
        {"price": 100.6, "features": {"last_price": 100.6}},
        {"price": 101.1, "features": {"last_price": 101.1}},
    ]

    label = rtc.event_followthrough_label_builder(checkpoints=(0.34, 0.67, 1.0), min_return=0.001)

    assert label(aligned, 0, 3) == 1.0
    assert label(whipsaw, 0, 3) is None


def test_abstain_quality_label_builder_marks_flat_high_friction_windows() -> None:
    sequence = [
        {
            "price": 100.0,
            "features": {
                "last_price": 100.0,
                "spread_bps": 26.0,
                "market_micro_tradeability_score_norm": 0.18,
                "execution_fitness_norm": 0.20,
                "market_micro_trade_halt_norm": 0.55,
            },
        },
        {"price": 100.04, "features": {"last_price": 100.04}},
        {"price": 100.05, "features": {"last_price": 100.05}},
    ]

    label = rtc.abstain_quality_label_builder(max_abs_return=0.001, min_stress_score=0.45)

    assert label(sequence, 0, 2) == 1.0


def test_regime_specific_label_builder_requires_matching_trend_context() -> None:
    trend_sequence = [
        {
            "price": 100.0,
            "features": {
                "last_price": 100.0,
                "day_regime_trend_norm": 0.82,
                "market_micro_trend_persistence_norm": 0.74,
                "pct_from_close": 0.004,
            },
        },
        {"price": 100.6, "features": {"last_price": 100.6}},
        {"price": 101.3, "features": {"last_price": 101.3}},
    ]
    fade_sequence = [
        {
            "price": 100.0,
            "features": {
                "last_price": 100.0,
                "day_regime_trend_norm": 0.84,
                "market_micro_trend_persistence_norm": 0.72,
                "pct_from_close": 0.004,
            },
        },
        {"price": 99.8, "features": {"last_price": 99.8}},
        {"price": 99.5, "features": {"last_price": 99.5}},
    ]

    label = rtc.regime_specific_label_builder(regime="trend", min_return=0.001, regime_threshold=0.60)

    assert label(trend_sequence, 0, 2) == 1.0
    assert label(fade_sequence, 0, 2) is None


def test_income_total_return_label_builder_rewards_income_plus_price_edge() -> None:
    sequence = [
        {
            "price": 100.0,
            "features": {
                "last_price": 100.0,
                "dividend_income_quality_norm": 0.78,
                "dividend_compounding_quality_norm": 0.74,
                "dividend_payout_stress_forward_norm": 0.22,
                "dividend_yield_norm": 0.84,
            },
        },
        {"price": 100.12, "features": {"last_price": 100.12}},
        {"price": 100.48, "features": {"last_price": 100.48}},
    ]

    label = rtc.income_total_return_label_builder(min_total_return=0.001)

    assert label(sequence, 0, 2) == 1.0


def test_derivatives_structure_label_builder_requires_flow_alignment() -> None:
    aligned = [
        {
            "price": 100.0,
            "features": {
                "last_price": 100.0,
                "options_net_call_premium_bias_norm": 0.74,
                "options_gamma_expiry_skew_norm": 0.70,
                "options_surface_change_norm": 0.68,
                "futures_order_book_imbalance_norm": 0.72,
                "futures_basis_bps_norm": 0.66,
                "futures_term_structure_norm": 0.64,
                "flow_direction_signed": 0.35,
                "core_options_structure_edge_norm": 0.72,
            },
        },
        {"price": 100.35, "features": {"last_price": 100.35}},
        {"price": 100.92, "features": {"last_price": 100.92}},
    ]
    conflicted = [
        {
            "price": 100.0,
            "features": {
                "last_price": 100.0,
                "options_net_call_premium_bias_norm": 0.74,
                "options_gamma_expiry_skew_norm": 0.70,
                "options_surface_change_norm": 0.68,
                "futures_order_book_imbalance_norm": 0.72,
                "futures_basis_bps_norm": 0.66,
                "futures_term_structure_norm": 0.64,
                "flow_direction_signed": 0.35,
                "core_options_structure_edge_norm": 0.72,
            },
        },
        {"price": 99.85, "features": {"last_price": 99.85}},
        {"price": 99.40, "features": {"last_price": 99.40}},
    ]

    label = rtc.derivatives_structure_label_builder(min_return=0.001)

    assert label(aligned, 0, 2) == 1.0
    assert label(conflicted, 0, 2) is None


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


def test_make_runtime_windowed_dataset_rebalances_extreme_label_skew(tmp_path) -> None:
    base_ts = datetime.now(timezone.utc)
    path = tmp_path / "decision_explanations" / "shadow_aggressive_equities" / "decision_explanations_20260325.jsonl"
    rows = []
    price = 100.0
    for i in range(96):
        if i > 0:
            price += (-0.75 if i % 12 == 0 else 0.28)
        prev_close = max(price - (0.28 if i % 12 else -0.75), 1e-8) if i > 0 else price
        rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(seconds=60 * i)).isoformat(),
                "mode": "shadow_aggressive_equities",
                "symbol": "SPY",
                "strategy": "grand_master_bot",
                "features": {
                    "last_price": price,
                    "pct_from_close": (price / max(prev_close, 1e-8)) - 1.0,
                    "vol_30m": 0.0035,
                },
                "metadata": {"layer": "grand_master", "snapshot_id": f"snap-{i}"},
            }
        )
    _write_jsonl(path, rows)

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)
    X, y, meta = rtc.make_runtime_windowed_dataset(
        sequences=sequences,
        feature_builder=lambda seq, idx: np.asarray([rtc.observation_feature(seq[idx], "pct_from_close")], dtype=np.float32),
        label_builder=rtc.direction_label_builder(min_return=0.0),
        window=2,
        horizon=1,
    )

    assert X.shape[0] == meta["sample_count"]
    assert y.shape[0] == meta["sample_count"]
    assert meta["label_balance_applied"] is True
    assert meta["label_balance_original_sample_count"] > meta["sample_count"]
    assert float(meta["label_balance_original_positive_rate"]) > 0.85
    assert float(meta["positive_rate"]) <= 0.8001


def test_make_runtime_windowed_dataset_caps_symbols_and_builds_label_audit(tmp_path, monkeypatch) -> None:
    base_ts = datetime.now(timezone.utc)
    spy_path = tmp_path / "decision_explanations" / "shadow_intraday_aggressive_equities" / "decision_explanations_20260325.jsonl"
    iwm_path = tmp_path / "decision_explanations" / "shadow_intraday_aggressive_equities" / "decision_explanations_20260326.jsonl"
    spy_rows = []
    iwm_rows = []
    spy_price = 100.0
    iwm_price = 50.0
    for i in range(64):
        if i > 0:
            spy_price += 0.45 if i % 2 == 0 else -0.30
        spy_rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(seconds=60 * i)).isoformat(),
                "mode": "shadow_intraday_aggressive_equities",
                "symbol": "SPY",
                "strategy": "grand_master_bot",
                "features": {
                    "last_price": spy_price,
                    "pct_from_close": 0.002 if i % 2 == 0 else -0.0015,
                    "day_regime_trend_norm": 0.82,
                    "market_micro_trend_persistence_norm": 0.76,
                },
                "metadata": {"layer": "grand_master", "snapshot_id": f"spy-{i}"},
            }
        )
    for i in range(20):
        if i > 0:
            iwm_price += 0.30 if i % 2 == 0 else -0.25
        iwm_rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(seconds=60 * (i + 80))).isoformat(),
                "mode": "shadow_intraday_aggressive_equities",
                "symbol": "IWM",
                "strategy": "grand_master_bot",
                "features": {
                    "last_price": iwm_price,
                    "pct_from_close": 0.002 if i % 2 == 0 else -0.001,
                    "news_shock_rate": 0.84,
                    "market_micro_range_expansion_norm": 0.74,
                },
                "metadata": {"layer": "grand_master", "snapshot_id": f"iwm-{i}"},
            }
        )
    _write_jsonl(spy_path, spy_rows)
    _write_jsonl(iwm_path, iwm_rows)
    monkeypatch.setenv("RUNTIME_TRAIN_SYMBOL_MAX_SHARE", "0.40")
    monkeypatch.setenv("RUNTIME_TRAIN_SYMBOL_CAP_MIN_SAMPLES", "16")
    monkeypatch.setenv("RUNTIME_TRAIN_REGIME_MAX_RATIO", "1.0")
    monkeypatch.setenv("RUNTIME_TRAIN_REGIME_BALANCE_MIN_SAMPLES", "16")

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)
    X, y, meta = rtc.make_runtime_windowed_dataset(
        sequences=sequences,
        feature_builder=lambda seq, idx: np.asarray([rtc.observation_feature(seq[idx], "pct_from_close")], dtype=np.float32),
        label_builder=rtc.direction_label_builder(min_return=0.0),
        window=2,
        horizon=1,
    )

    assert X.shape[0] == y.shape[0] == meta["sample_count"]
    assert meta["symbol_cap_applied"] is True
    assert meta["regime_balance_applied"] is True
    assert meta["label_audit"]["by_symbol"][0]["name"] in {"SPY", "IWM"}
    family_names = {row["name"] for row in meta["label_audit"]["by_family"]}
    assert "intraday" in family_names
    regime_names = {row["name"] for row in meta["label_audit"]["by_regime"]}
    assert {"trend", "shock"}.issubset(regime_names)


def test_make_runtime_windowed_dataset_applies_memory_sample_cap(tmp_path) -> None:
    base_ts = datetime.now(timezone.utc)
    path = tmp_path / "decision_explanations" / "shadow_intraday_aggressive_equities" / "decision_explanations_20260327.jsonl"
    rows = []
    price = 100.0
    for i in range(48):
        if i > 0:
            price += 0.35 if i % 2 == 0 else -0.20
        rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(seconds=60 * i)).isoformat(),
                "mode": "shadow_intraday_aggressive_equities",
                "symbol": "SPY",
                "strategy": "grand_master_bot",
                "features": {
                    "last_price": price,
                    "pct_from_close": 0.002 if i % 2 == 0 else -0.0015,
                },
                "metadata": {"layer": "grand_master", "snapshot_id": f"snap-{i}"},
            }
        )
    _write_jsonl(path, rows)

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)
    X, y, meta = rtc.make_runtime_windowed_dataset(
        sequences=sequences,
        feature_builder=lambda seq, idx: np.asarray([rtc.observation_feature(seq[idx], "pct_from_close")], dtype=np.float32),
        label_builder=rtc.direction_label_builder(min_return=0.0),
        max_samples=6,
        window=2,
        horizon=1,
    )

    assert X.shape[0] == 6
    assert y.shape[0] == 6
    assert meta["memory_sample_cap_applied"] is True
    assert meta["memory_sample_cap_limit"] == 6
    assert meta["memory_sample_cap_original_count"] > 6


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
                "sector_rotation_score": 0.014,
                "sector_advancers": 8,
                "sector_decliners": 3,
                "sector_leader_strength": 0.024,
                "sector_laggard_strength": -0.019,
            }
        ),
        encoding="utf-8",
    )
    (external_root / "market_micro_latest.json").write_text(
        json.dumps(
            {
                "derived": {
                    "global_features": {
                        "market_micro_opening_auction_imbalance_norm": 0.62,
                        "market_micro_opening_drive_pressure_norm": 0.71,
                        "market_micro_spread_regime_norm": 0.39,
                        "market_micro_quote_fade_rate_norm": 0.27,
                    },
                    "symbol_features": {
                        "TLT": {
                            "market_micro_closing_cross_pressure_norm": 0.68,
                            "market_micro_queue_depth_decay_norm": 0.31,
                            "etf_fund_family_creation_pressure_norm": 0.56,
                        }
                    },
                }
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
    (external_root / "schwab_education_context_latest.json").write_text(
        json.dumps(
            {
                "derived": {
                    "news_features": {
                        "news_source_quality_norm": 0.97,
                        "news_after_hours_norm": 0.77,
                    },
                    "global_features": {
                        "schwab_education_item_density_norm": 0.68,
                        "schwab_education_recent_activity_norm": 0.74,
                        "schwab_education_symbol_coverage_norm": 0.33,
                    },
                    "symbol_features": {
                        "TLT": {
                            "news_available": 0.9,
                            "news_entity_relevance_norm": 0.92,
                            "schwab_education_symbol_frequency_norm": 0.72,
                            "schwab_education_symbol_recency_norm": 0.88,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (external_root / "sec_edgar_latest.json").write_text(
        json.dumps(
            {
                "derived": {
                    "news_features": {
                        "news_source_quality_norm": 0.96,
                        "news_topic_earnings_norm": 0.72,
                    },
                    "calendar_features": {
                        "calendar_events_24h_norm": 0.25,
                    },
                    "global_features": {
                        "sec_recent_symbols_norm": 0.45,
                        "sec_estimate_revision_drift_norm": 0.77,
                    },
                    "symbol_features": {
                        "TLT": {
                            "sec_guidance_7d_norm": 0.55,
                            "sec_recent_proximity_norm": 0.81,
                            "sec_insider_buy_30d_norm": 0.66,
                            "sec_earnings_whisper_surprise_norm": 0.73,
                            "sec_split_hazard_30d_norm": 0.48,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (external_root / "extended_quant_context_latest.json").write_text(
        json.dumps(
            {
                "derived": {
                    "calendar_features": {
                        "calendar_opex_week_norm": 0.88,
                        "calendar_futures_roll_window_norm": 0.74,
                    },
                    "global_features": {
                        "sofr_funding_stress_norm": 0.66,
                        "cboe_put_call_stress_norm": 0.58,
                    },
                    "symbol_features": {
                        "TLT": {
                            "short_threshold_listed_norm": 1.0,
                        }
                    },
                    "bond_reference_overlay": {
                        "funding_stress_norm": 0.66,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (external_root / "options_flow_context_latest.json").write_text(
        json.dumps(
            {
                "derived": {
                    "global_features": {
                        "short_borrow_fee_norm": 0.32,
                        "tasty_dealer_gamma_pressure_norm": 0.61,
                        "options_iv_term_structure_norm": 0.64,
                    },
                    "symbol_features": {
                        "TLT": {
                            "short_borrow_availability_norm": 0.84,
                            "tasty_max_pain_proximity_norm": 0.57,
                            "options_iv_skew_norm": 0.42,
                            "options_gamma_expiry_skew_norm": 0.61,
                            "options_vol_regime_norm": 0.58,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (external_root / "crypto_market_context_latest.json").write_text(
        json.dumps(
            {
                "derived": {
                    "news_features": {
                        "news_available": 0.85,
                        "news_items_24h": 0.65,
                        "news_sentiment": -0.28,
                        "news_shock_rate": 0.44,
                        "news_source_quality_norm": 0.81,
                    },
                    "global_features": {
                        "crypto_cross_provider_price_agreement_norm": 0.91,
                        "crypto_defillama_stablecoin_growth_norm": 0.64,
                    },
                    "symbol_features": {
                        "TLT": {
                            "crypto_deribit_mark_iv_norm": 0.0,
                        },
                        "BTC-USD": {
                            "crypto_deribit_mark_iv_norm": 0.74,
                            "crypto_hyperliquid_funding_norm": 0.58,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (external_root / "market_crypto_correlation_latest.json").write_text(
        json.dumps(
            {
                "derived": {
                    "global_features": {
                        "market_crypto_risk_corr_norm": 0.72,
                        "market_crypto_corr_confidence_norm": 0.61,
                    },
                    "symbol_features": {
                        "TLT": {
                            "market_crypto_tlt_corr_norm": 0.33,
                            "market_crypto_current_alignment_norm": 0.57,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (external_root / "dividend_drip_state_latest.json").write_text(
        json.dumps(
            {
                "derived": {
                    "global_features": {
                        "dividend_drip_active_norm": 0.44,
                        "dividend_drip_confidence_norm": 0.71,
                    },
                    "symbol_features": {
                        "TLT": {
                            "dividend_drip_active_norm": 0.0,
                        },
                        "SCHD": {
                            "dividend_drip_active_norm": 0.83,
                            "dividend_drip_recent_reinvest_norm": 0.64,
                            "dividend_drip_cash_only_norm": 0.12,
                            "dividend_drip_share_credit_norm": 0.58,
                            "dividend_drip_event_recency_norm": 0.91,
                            "dividend_drip_confidence_norm": 0.88,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)
    row = sequences[("shadow_bond_equities", "TLT")][0]
    features = row["features"]

    assert features["calendar_treasury_auction_norm"] == 0.72
    assert features["calendar_opex_week_norm"] == 0.88
    assert features["breadth_risk_off_norm"] > 0.0
    assert features["breadth_sector_rotation_norm"] > 0.0
    assert features["breadth_leader_laggard_spread_norm"] > 0.0
    assert features["bond_yield_10y_norm"] > 0.0
    assert features["news_available"] > 0.0
    assert features["news_sentiment"] < 0.0
    assert features["news_shock_rate"] >= 0.44
    assert features["news_after_hours_norm"] == 0.77
    assert features["news_entity_relevance_norm"] == 0.92
    assert features["sec_guidance_7d_norm"] == 0.55
    assert features["sec_insider_buy_30d_norm"] == 0.66
    assert features["sec_earnings_whisper_surprise_norm"] == 0.73
    assert features["sec_estimate_revision_drift_norm"] == 0.77
    assert features["schwab_education_item_density_norm"] == 0.68
    assert features["schwab_education_symbol_frequency_norm"] == 0.72
    assert features["sofr_funding_stress_norm"] == 0.66
    assert features["short_threshold_listed_norm"] == 1.0
    assert features["short_borrow_availability_norm"] == 0.84
    assert features["short_borrow_fee_norm"] == 0.32
    assert features["tasty_dealer_gamma_pressure_norm"] == 0.61
    assert features["tasty_max_pain_proximity_norm"] == 0.57
    assert features["options_iv_skew_norm"] == 0.42
    assert features["options_iv_term_structure_norm"] == 0.64
    assert features["options_vol_regime_norm"] == 0.58
    assert features["market_micro_opening_drive_pressure_norm"] == 0.71
    assert features["market_micro_closing_cross_pressure_norm"] == 0.68
    assert features["etf_fund_family_creation_pressure_norm"] == 0.56
    assert features["crypto_cross_provider_price_agreement_norm"] == 0.91
    assert features["crypto_defillama_stablecoin_growth_norm"] == 0.64
    assert features["market_crypto_risk_corr_norm"] == 0.72
    assert features["market_crypto_tlt_corr_norm"] == 0.33
    assert features["market_crypto_current_alignment_norm"] == 0.57
    assert features["dividend_drip_active_norm"] == 0.0
    assert features["dividend_drip_confidence_norm"] == 0.71


def test_load_runtime_observation_sequences_backfills_dividend_drip_state(tmp_path) -> None:
    ts = datetime.now(timezone.utc)
    path = tmp_path / "decision_explanations" / "shadow_dividend_equities" / "decision_explanations_20260318.jsonl"

    _write_jsonl(
        path,
        [
            {
                "timestamp_utc": ts.isoformat(),
                "mode": "shadow_dividend_equities",
                "symbol": "SCHD",
                "strategy": "grand_master_bot",
                "features": {
                    "last_price": 27.5,
                    "pct_from_close": 0.004,
                    "vol_30m": 0.002,
                },
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-drip-1"},
            }
        ],
    )

    external_root = tmp_path / "data" / "external_context"
    external_root.mkdir(parents=True, exist_ok=True)
    (external_root / "dividend_drip_state_latest.json").write_text(
        json.dumps(
            {
                "derived": {
                    "global_features": {
                        "dividend_drip_active_norm": 0.52,
                        "dividend_drip_confidence_norm": 0.61,
                    },
                    "symbol_features": {
                        "SCHD": {
                            "dividend_drip_active_norm": 0.86,
                            "dividend_drip_recent_reinvest_norm": 0.72,
                            "dividend_drip_cash_only_norm": 0.14,
                            "dividend_drip_share_credit_norm": 0.63,
                            "dividend_drip_event_recency_norm": 0.93,
                            "dividend_drip_confidence_norm": 0.89,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    sequences = rtc.load_runtime_observation_sequences(tmp_path, lookback_days=2)
    row = sequences[("shadow_dividend_equities", "SCHD")][0]
    features = row["features"]

    assert features["dividend_drip_active_norm"] == 0.86
    assert features["dividend_drip_recent_reinvest_norm"] == 0.72
    assert features["dividend_drip_cash_only_norm"] == 0.14
    assert features["dividend_drip_share_credit_norm"] == 0.63
    assert features["dividend_drip_event_recency_norm"] == 0.93
    assert features["dividend_drip_confidence_norm"] == 0.89


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
