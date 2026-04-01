from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import collect_market_crypto_correlation_context as corr_ctx


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_collect_market_crypto_correlation_context_joins_stock_and_crypto_roots(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    external_root = tmp_path / "external"
    base_ts = datetime(2026, 3, 22, 13, 0, tzinfo=timezone.utc)
    btc_path = project_root / "governance" / "shadow_crypto" / "master_control_20260322.jsonl"
    bond_path = external_root / "governance" / "shadow_bond_equities" / "master_control_20260322.jsonl"

    crypto_moves = [-0.018, -0.012, -0.006, -0.001, 0.004, 0.010, 0.016, 0.021]
    stock_moves = [-0.015, -0.010, -0.004, -0.001, 0.003, 0.009, 0.013, 0.018]
    tlt_moves = [0.010, 0.007, 0.003, 0.001, -0.002, -0.006, -0.010, -0.013]
    uup_moves = [0.012, 0.008, 0.004, 0.001, -0.002, -0.007, -0.011, -0.014]
    gld_moves = [-0.008, -0.006, -0.003, -0.001, 0.001, 0.005, 0.007, 0.010]

    crypto_rows = []
    bond_rows = []
    for i, (btc_move, spy_move, tlt_move, uup_move, gld_move) in enumerate(
        zip(crypto_moves, stock_moves, tlt_moves, uup_moves, gld_moves)
    ):
        ts = (base_ts + timedelta(minutes=5 * i)).isoformat()
        crypto_rows.append(
            {
                "timestamp_utc": ts,
                "symbol": "BTC-USD",
                "broker": "coinbase",
                "market": {
                    "last_price": 70000.0 * (1.0 + btc_move),
                    "pct_from_close": btc_move,
                    "mom_5m": btc_move * 0.6,
                    "return_1m": btc_move * 0.18,
                },
                "context_market": {
                    "ETH-USD": {
                        "last_price": 2000.0 * (1.0 + (btc_move * 0.95)),
                        "pct_from_close": btc_move * 0.95,
                        "mom_5m": btc_move * 0.57,
                        "return_1m": btc_move * 0.17,
                    },
                    "SOL-USD": {
                        "last_price": 120.0 * (1.0 + (btc_move * 1.08)),
                        "pct_from_close": btc_move * 1.08,
                        "mom_5m": btc_move * 0.62,
                        "return_1m": btc_move * 0.19,
                    },
                },
            }
        )
        bond_rows.append(
            {
                "timestamp_utc": ts,
                "symbol": "HYG",
                "broker": "schwab",
                "market": {
                    "last_price": 80.0 * (1.0 + (spy_move * 0.8)),
                    "pct_from_close": spy_move * 0.8,
                    "mom_5m": spy_move * 0.45,
                    "return_1m": spy_move * 0.12,
                },
                "context_market": {
                    "SPY": {
                        "last_price": 500.0 * (1.0 + spy_move),
                        "pct_from_close": spy_move,
                        "mom_5m": spy_move * 0.65,
                        "return_1m": spy_move * 0.21,
                    },
                    "QQQ": {
                        "last_price": 430.0 * (1.0 + (spy_move * 1.04)),
                        "pct_from_close": spy_move * 1.04,
                        "mom_5m": spy_move * 0.69,
                        "return_1m": spy_move * 0.22,
                    },
                    "TLT": {
                        "last_price": 95.0 * (1.0 + tlt_move),
                        "pct_from_close": tlt_move,
                        "mom_5m": tlt_move * 0.72,
                        "return_1m": tlt_move * 0.18,
                    },
                    "UUP": {
                        "last_price": 28.0 * (1.0 + uup_move),
                        "pct_from_close": uup_move,
                        "mom_5m": uup_move * 0.70,
                        "return_1m": uup_move * 0.16,
                    },
                    "GLD": {
                        "last_price": 210.0 * (1.0 + gld_move),
                        "pct_from_close": gld_move,
                        "mom_5m": gld_move * 0.68,
                        "return_1m": gld_move * 0.15,
                    },
                },
            }
        )

    _write_jsonl(btc_path, crypto_rows)
    _write_jsonl(bond_path, bond_rows)

    original_market_hours = corr_ctx._is_us_equity_market_hours
    corr_ctx._is_us_equity_market_hours = lambda now: False
    try:
        payload, status = corr_ctx.collect_market_crypto_correlation_context(
            project_root=project_root,
            lookback_days=3,
            bucket_seconds=300,
            min_points=6,
            extra_roots=[external_root],
        )
    finally:
        corr_ctx._is_us_equity_market_hours = original_market_hours

    assert status["ok"] is True
    assert status["aligned_pairs"] >= 3
    assert payload["derived"]["global_features"]["market_crypto_risk_corr_norm"] > 0.9
    assert payload["derived"]["global_features"]["market_crypto_uup_inverse_corr_norm"] > 0.9
    assert payload["derived"]["symbol_features"]["BTC-USD"]["market_crypto_spy_corr_norm"] > 0.9
    assert payload["derived"]["symbol_features"]["BTC-USD"]["market_crypto_corr_confidence_norm"] > 0.2


def test_collect_market_crypto_correlation_context_keeps_live_crypto_jsonl_when_stock_csv_exists(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    external_root = tmp_path / "external"
    crypto_path = project_root / "governance" / "shadow_crypto" / "master_control_20260323.jsonl"
    stock_csv_path = external_root / "exports" / "csv" / "master_control_20260323.csv"
    base_ts = datetime(2026, 3, 23, 14, 0, tzinfo=timezone.utc)

    crypto_rows = []
    stock_csv_rows = []
    for i, move in enumerate([-0.015, -0.010, -0.006, -0.001, 0.003, 0.008, 0.012, 0.017]):
        ts = (base_ts + timedelta(minutes=5 * i)).isoformat()
        crypto_rows.append(
            {
                "timestamp_utc": ts,
                "symbol": "BTC-USD",
                "market": {
                    "last_price": 70000.0 * (1.0 + move),
                    "pct_from_close": move,
                    "mom_5m": move * 0.62,
                    "return_1m": move * 0.19,
                },
                "context_market": {
                    "ETH-USD": {
                        "last_price": 2000.0 * (1.0 + (move * 0.93)),
                        "pct_from_close": move * 0.93,
                        "mom_5m": move * 0.58,
                        "return_1m": move * 0.18,
                    }
                },
            }
        )
        stock_csv_rows.append(
            {
                "timestamp_utc": ts,
                "symbol": "SPY",
                "market": json.dumps(
                    {
                        "last_price": 500.0 * (1.0 + (move * 0.88)),
                        "pct_from_close": move * 0.88,
                        "mom_5m": move * 0.54,
                        "return_1m": move * 0.17,
                    }
                ),
                "context_market": json.dumps(
                    {
                        "QQQ": {
                            "last_price": 430.0 * (1.0 + (move * 0.95)),
                            "pct_from_close": move * 0.95,
                            "mom_5m": move * 0.57,
                            "return_1m": move * 0.18,
                        },
                        "TLT": {
                            "last_price": 95.0 * (1.0 - (move * 0.40)),
                            "pct_from_close": -(move * 0.40),
                            "mom_5m": -(move * 0.24),
                            "return_1m": -(move * 0.10),
                        },
                    }
                ),
            }
        )

    _write_jsonl(crypto_path, crypto_rows)
    _write_csv(stock_csv_path, stock_csv_rows)

    original_market_hours = corr_ctx._is_us_equity_market_hours
    corr_ctx._is_us_equity_market_hours = lambda now: False
    try:
        payload, status = corr_ctx.collect_market_crypto_correlation_context(
            project_root=project_root,
            lookback_days=3,
            bucket_seconds=300,
            min_points=6,
            extra_roots=[external_root],
        )
    finally:
        corr_ctx._is_us_equity_market_hours = original_market_hours

    assert status["mode"] == "exact"
    assert status["exact_aligned_pairs"] >= 1
    assert payload["derived"]["pair_metrics"][0]["mode"] == "exact"
    assert payload["derived"]["symbol_features"]["BTC-USD"]["market_crypto_corr_confidence_norm"] > 0.2


def test_collect_market_crypto_correlation_context_carries_forward_last_usable_overlap(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    external_root = tmp_path / "external"
    base_ts = datetime(2026, 3, 21, 20, 0, tzinfo=timezone.utc)
    btc_path = project_root / "governance" / "shadow_crypto" / "master_control_20260321.jsonl"
    stock_path = external_root / "governance" / "shadow_aggressive_equities" / "master_control_20260321.jsonl"

    crypto_moves = [-0.016, -0.010, -0.005, 0.000, 0.004, 0.009, 0.014, 0.019]
    stock_moves = [-0.013, -0.008, -0.003, 0.000, 0.003, 0.007, 0.011, 0.015]

    crypto_rows = []
    stock_rows = []
    for i, (btc_move, spy_move) in enumerate(zip(crypto_moves, stock_moves)):
        ts = (base_ts + timedelta(minutes=5 * i)).isoformat()
        crypto_rows.append(
            {
                "timestamp_utc": ts,
                "symbol": "BTC-USD",
                "market": {
                    "last_price": 70000.0 * (1.0 + btc_move),
                    "pct_from_close": btc_move,
                    "mom_5m": btc_move * 0.5,
                    "return_1m": btc_move * 0.2,
                },
            }
        )
        stock_rows.append(
            {
                "timestamp_utc": ts,
                "symbol": "SPY",
                "market": {
                    "last_price": 500.0 * (1.0 + spy_move),
                    "pct_from_close": spy_move,
                    "mom_5m": spy_move * 0.5,
                    "return_1m": spy_move * 0.2,
                },
                "context_market": {
                    "QQQ": {
                        "last_price": 430.0 * (1.0 + (spy_move * 1.04)),
                        "pct_from_close": spy_move * 1.04,
                        "mom_5m": spy_move * 0.52,
                        "return_1m": spy_move * 0.21,
                    }
                },
            }
        )

    _write_jsonl(btc_path, crypto_rows)
    _write_jsonl(stock_path, stock_rows)

    _, first_status = corr_ctx.collect_market_crypto_correlation_context(
        project_root=project_root,
        lookback_days=3,
        bucket_seconds=300,
        min_points=6,
        extra_roots=[external_root],
    )

    assert first_status["mode"] == "exact"
    assert first_status["aligned_pairs"] >= 1

    stock_path.unlink()

    _, second_status = corr_ctx.collect_market_crypto_correlation_context(
        project_root=project_root,
        lookback_days=3,
        bucket_seconds=300,
        min_points=6,
        extra_roots=[external_root],
    )

    cache_payload = json.loads(
        (project_root / "exports" / "external_context" / "market_crypto_correlation_cache_latest.json").read_text(encoding="utf-8")
    )

    assert second_status["mode"] == "carry_forward_last_usable"
    assert second_status["aligned_pairs"] >= 1
    assert second_status["exact_aligned_pairs"] == 0
    assert "carry_forward_last_usable:using_cached_overlap" in second_status["warnings"]
    assert cache_payload["last_usable"]["aligned_pairs"] >= 1


def test_collect_market_crypto_correlation_context_uses_approximate_overlap_when_buckets_do_not_match(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    external_root = tmp_path / "external"
    crypto_path = project_root / "governance" / "shadow_crypto" / "master_control_20260322.jsonl"
    stock_path = external_root / "governance" / "shadow_aggressive_equities" / "master_control_20260322.jsonl"
    base_ts = datetime(2026, 3, 22, 14, 0, tzinfo=timezone.utc)

    crypto_rows = []
    stock_rows = []
    for i, move in enumerate([-0.012, -0.008, -0.003, 0.001, 0.004, 0.009, 0.013, 0.017]):
        crypto_rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(minutes=60 * i)).isoformat(),
                "symbol": "BTC-USD",
                "market": {
                    "last_price": 70000.0 * (1.0 + move),
                    "pct_from_close": move,
                    "mom_5m": move * 0.6,
                    "return_1m": move * 0.2,
                },
            }
        )
        stock_rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(minutes=(30 * i) + 11)).isoformat(),
                "symbol": "SPY",
                "market": {
                    "last_price": 500.0 * (1.0 + (move * 0.92)),
                    "pct_from_close": move * 0.92,
                    "mom_5m": move * 0.55,
                    "return_1m": move * 0.18,
                },
                "context_market": {
                    "QQQ": {
                        "last_price": 430.0 * (1.0 + (move * 0.97)),
                        "pct_from_close": move * 0.97,
                        "mom_5m": move * 0.57,
                        "return_1m": move * 0.19,
                    }
                },
            }
        )

    _write_jsonl(crypto_path, crypto_rows)
    _write_jsonl(stock_path, stock_rows)

    original_market_hours = corr_ctx._is_us_equity_market_hours
    corr_ctx._is_us_equity_market_hours = lambda now: False
    try:
        payload, status = corr_ctx.collect_market_crypto_correlation_context(
            project_root=project_root,
            lookback_days=3,
            bucket_seconds=300,
            min_points=6,
            extra_roots=[external_root],
        )
    finally:
        corr_ctx._is_us_equity_market_hours = original_market_hours

    assert status["mode"] == "approximate_overlap"
    assert status["aligned_pairs"] >= 1
    assert status["exact_aligned_pairs"] == 0
    assert "approximate_overlap_pairs:" in " ".join(status["warnings"])
    assert payload["derived"]["pair_metrics"][0]["mode"] == "asof"


def test_collect_market_crypto_correlation_context_limits_asof_gap_during_market_hours(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    external_root = tmp_path / "external"
    crypto_path = project_root / "governance" / "shadow_crypto" / "master_control_20260323.jsonl"
    stock_path = external_root / "governance" / "shadow_aggressive_equities" / "master_control_20260323.jsonl"
    base_ts = datetime(2026, 3, 23, 14, 0, tzinfo=timezone.utc)

    crypto_rows = []
    stock_rows = []
    for i, move in enumerate([-0.012, -0.008, -0.003, 0.001, 0.004, 0.009, 0.013, 0.017]):
        crypto_rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(minutes=30 * i)).isoformat(),
                "symbol": "BTC-USD",
                "market": {
                    "last_price": 70000.0 * (1.0 + move),
                    "pct_from_close": move,
                    "mom_5m": move * 0.6,
                    "return_1m": move * 0.2,
                },
            }
        )
        stock_rows.append(
            {
                "timestamp_utc": (base_ts + timedelta(minutes=(60 * i) + 31)).isoformat(),
                "symbol": "SPY",
                "market": {
                    "last_price": 500.0 * (1.0 + (move * 0.92)),
                    "pct_from_close": move * 0.92,
                    "mom_5m": move * 0.55,
                    "return_1m": move * 0.18,
                },
                "context_market": {
                    "QQQ": {
                        "last_price": 430.0 * (1.0 + (move * 0.97)),
                        "pct_from_close": move * 0.97,
                        "mom_5m": move * 0.57,
                        "return_1m": move * 0.19,
                    }
                },
            }
        )

    _write_jsonl(crypto_path, crypto_rows)
    _write_jsonl(stock_path, stock_rows)

    original_market_hours = corr_ctx._is_us_equity_market_hours
    corr_ctx._is_us_equity_market_hours = lambda now: True
    try:
        payload, status = corr_ctx.collect_market_crypto_correlation_context(
            project_root=project_root,
            lookback_days=3,
            bucket_seconds=300,
            min_points=6,
            extra_roots=[external_root],
        )
    finally:
        corr_ctx._is_us_equity_market_hours = original_market_hours

    assert status["market_hours_bias"] is True
    assert status["asof_max_lag_seconds"] == 900
    assert status["aligned_pairs"] == 0
    assert status["mode"] == "exact"
    assert "approximate_overlap_pairs:" not in " ".join(status["warnings"])


def test_collect_market_crypto_correlation_context_reuses_cached_result_when_inputs_unchanged(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    external_root = tmp_path / "external"
    crypto_path = project_root / "governance" / "shadow_crypto" / "master_control_20260324.jsonl"
    stock_path = external_root / "governance" / "shadow_aggressive_equities" / "master_control_20260324.jsonl"
    base_ts = datetime(2026, 3, 24, 14, 0, tzinfo=timezone.utc)

    crypto_rows = []
    stock_rows = []
    for i, move in enumerate([-0.014, -0.009, -0.004, 0.001, 0.005, 0.010, 0.014, 0.018]):
        ts = (base_ts + timedelta(minutes=5 * i)).isoformat()
        crypto_rows.append(
            {
                "timestamp_utc": ts,
                "symbol": "BTC-USD",
                "market": {
                    "last_price": 70000.0 * (1.0 + move),
                    "pct_from_close": move,
                    "mom_5m": move * 0.6,
                    "return_1m": move * 0.2,
                },
                "context_market": {
                    "ETH-USD": {
                        "last_price": 2000.0 * (1.0 + (move * 0.95)),
                        "pct_from_close": move * 0.95,
                        "mom_5m": move * 0.57,
                        "return_1m": move * 0.19,
                    }
                },
            }
        )
        stock_rows.append(
            {
                "timestamp_utc": ts,
                "symbol": "SPY",
                "market": {
                    "last_price": 500.0 * (1.0 + (move * 0.90)),
                    "pct_from_close": move * 0.90,
                    "mom_5m": move * 0.55,
                    "return_1m": move * 0.18,
                },
                "context_market": {
                    "QQQ": {
                        "last_price": 430.0 * (1.0 + (move * 0.95)),
                        "pct_from_close": move * 0.95,
                        "mom_5m": move * 0.58,
                        "return_1m": move * 0.19,
                    },
                    "TLT": {
                        "last_price": 95.0 * (1.0 - (move * 0.45)),
                        "pct_from_close": -(move * 0.45),
                        "mom_5m": -(move * 0.24),
                        "return_1m": -(move * 0.10),
                    },
                },
            }
        )

    _write_jsonl(crypto_path, crypto_rows)
    _write_jsonl(stock_path, stock_rows)

    _, first_status = corr_ctx.collect_market_crypto_correlation_context(
        project_root=project_root,
        lookback_days=3,
        bucket_seconds=300,
        min_points=6,
        extra_roots=[external_root],
    )
    _, second_status = corr_ctx.collect_market_crypto_correlation_context(
        project_root=project_root,
        lookback_days=3,
        bucket_seconds=300,
        min_points=6,
        extra_roots=[external_root],
    )

    assert first_status["cache_result_reused"] is False
    assert second_status["cache_result_reused"] is True
    assert second_status["aligned_pairs"] >= 1
