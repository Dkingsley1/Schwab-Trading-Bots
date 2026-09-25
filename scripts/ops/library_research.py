"""Bounded offline library comparison, never a broker or native bot decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_BYTES = 4 * 1024 * 1024
MAX_ROWS = 5000
TIMEOUT_SECONDS = 90
PACKAGES = (
    "TA-Lib",
    "backtrader",
    "vectorbt",
    "polars",
    "duckdb",
    "pyarrow",
    "msgspec",
    "pandas-ta-classic",
    "backtesting",
    "quantstats",
    "arch",
)
AUTHORITY = {
    "live_execution_authority": False,
    "autonomous_execution": False,
    "production_promotion_credit": False,
    "strategy_profitability_proven": False,
}


def validate_candles(payload, *, bar_seconds, now=None):
    if isinstance(bar_seconds, bool) or not 1 <= bar_seconds <= 31_536_000:
        raise ValueError("invalid_bar_seconds")
    rows = payload.get("candles") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not 40 <= len(rows) <= MAX_ROWS:
        raise ValueError("require_40_to_5000_candles")
    now = time.time() if now is None else now
    clean = []
    previous = -1
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid_candle")
        values = [
            row.get(k) for k in ("datetime", "open", "high", "low", "close", "volume")
        ]
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in values
        ):
            raise ValueError("invalid_candle_number")
        ts, opening, high, low, close, volume = values
        if ts != int(ts) or ts <= previous or ts < 0:
            raise ValueError("timestamps_must_be_ordered_unique_epoch_milliseconds")
        if ts / 1000 + bar_seconds > now:
            raise ValueError("unclosed_or_future_candle")
        if (
            not 0 < low <= min(opening, close) <= max(opening, close) <= high
            or volume < 0
        ):
            raise ValueError("invalid_ohlcv_range")
        clean.append(
            dict(zip(("datetime", "open", "high", "low", "close", "volume"), values))
        )
        previous = ts
    return clean


def fixture():
    rows = []
    for i in range(180):
        opening = 50 + math.sin(i / 9) * 4 + i / 150
        close = opening + math.sin(i / 3) * 0.3
        rows.append(
            {
                "datetime": (1_700_000_000 + i * 3600) * 1000,
                "open": opening,
                "high": max(opening, close) + 1,
                "low": min(opening, close) - 1,
                "close": close,
                "volume": 1000 + i,
            }
        )
    return {"candles": rows}


def compare(payload, *, bar_seconds, synthetic=False):
    """Run one fixed SMA example with identical next-open fills in two engines."""
    rows = validate_candles(payload, bar_seconds=bar_seconds)
    from importlib.metadata import version
    import backtrader as bt
    import duckdb
    import msgspec
    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq
    import talib
    import vectorbt as vbt

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    frame = pl.DataFrame(rows).with_columns(
        pl.col("close").rolling_mean(10).alias("sma10"),
        pl.col("close").rolling_mean(30).alias("sma30"),
    )
    close = np.asarray(frame["close"], dtype=float)
    high = np.asarray(frame["high"], dtype=float)
    low = np.asarray(frame["low"], dtype=float)
    fast, slow = talib.SMA(close, 10), talib.SMA(close, 30)
    indicator_parity = bool(
        np.allclose(fast, frame["sma10"].to_numpy(), equal_nan=True)
        and np.allclose(slow, frame["sma30"].to_numpy(), equal_nan=True)
    )
    wanted = np.isfinite(slow) & (fast > slow)
    data = pd.DataFrame(rows)
    data.index = pd.to_datetime(
        data.pop("datetime"), unit="ms", utc=True
    ).dt.tz_localize(None)
    cash, fee, slippage = 10_000.0, 0.001, 0.0003

    class ComparisonStrategy(bt.Strategy):
        def __init__(self):
            self.fills = []
            self.failures = []

        def next(self):
            # Decide at this close; Backtrader market orders fill next open.
            target = bool(wanted[len(self) - 1])
            if target and not self.position:
                self.buy(size=1)
            elif not target and self.position:
                self.sell(size=1)

        def notify_order(self, order):
            if order.status == order.Completed:
                self.fills.append(
                    (1 if order.isbuy() else -1, float(order.executed.price))
                )
            elif order.status in (order.Margin, order.Rejected, order.Canceled):
                self.failures.append(order.getstatusname())

    cerebro = bt.Cerebro(stdstats=False, maxcpus=1)
    cerebro.adddata(bt.feeds.PandasData(dataname=data))
    cerebro.addstrategy(ComparisonStrategy)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=fee)
    cerebro.broker.set_slippage_perc(
        slippage, slip_open=True, slip_match=True, slip_out=True
    )
    result = cerebro.run()[0]
    bt_value = float(cerebro.broker.getvalue())
    entries = np.r_[False, wanted[:-1]]
    exits = np.r_[False, ~wanted[:-1]]
    portfolio = vbt.Portfolio.from_signals(
        data["close"],
        entries,
        exits,
        price=data["open"],
        size=1,
        init_cash=cash,
        fees=fee,
        slippage=slippage,
        accumulate=False,
        direction="longonly",
        freq=f"{bar_seconds}s",
    )
    vbt_value = float(portfolio.final_value())
    order_records = portfolio.orders.records
    vbt_fills = [
        (1 if int(r.side) == 0 else -1, float(r.price))
        for r in order_records.itertuples()
    ]
    fill_parity = len(result.fills) == len(vbt_fills) and all(
        a[0] == b[0] and math.isclose(a[1], b[1], rel_tol=1e-10, abs_tol=1e-8)
        for a, b in zip(result.fills, vbt_fills)
    )
    engine_parity = (
        not result.failures
        and fill_parity
        and math.isclose(bt_value, vbt_value, abs_tol=1e-7, rel_tol=1e-10)
    )

    table = frame.to_arrow()
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    compressed = sink.getvalue()
    restored = pq.read_table(pa.BufferReader(compressed))
    with duckdb.connect(
        ":memory:",
        config={
            "threads": "1",
            "memory_limit": "128MB",
            "enable_external_access": "false",
        },
    ) as con:
        con.register("candles", restored)
        count, min_close, max_close = con.execute(
            "SELECT count(*), min(close), max(close) FROM candles"
        ).fetchone()
    archive_parity = restored.equals(table) and count == len(rows)
    codec_parity = msgspec.json.decode(msgspec.json.encode(rows)) == rows
    from scripts.ops.library_research_extensions import (
        indicator_diagnostics,
        timing_control,
        performance_diagnostics,
        volatility_diagnostics,
    )

    indicators = indicator_diagnostics(data, fast, slow)
    timing = timing_control(data, wanted, cash=cash, bar_seconds=bar_seconds)
    performance = performance_diagnostics(portfolio.value(), starting_cash=cash)
    volatility = volatility_diagnostics(data["close"])
    checks = {
        "polars_talib_sma_parity": indicator_parity,
        "backtrader_vectorbt_fill_and_equity_parity": bool(engine_parity),
        "parquet_restore_and_duckdb_count": bool(archive_parity),
        "msgspec_roundtrip": codec_parity,
        "pandas_ta_classic_native_sma_parity": indicators["sma_parity"],
        "pandas_ta_classic_sma_prefix_invariance": indicators["sma_prefix_invariance"],
        "backtesting_zero_cost_timing_parity": timing[
            "fill_timing_price_and_equity_parity"
        ],
        "quantstats_equity_reconstruction": performance["equity_reconstruction_parity"],
        "arch_fit_converged": volatility["converged"],
    }
    return {
        "purpose": "offline_library_comparison",
        "ok": all(value is not False for value in checks.values()),
        "data_kind": (
            "synthetic_fixture" if synthetic else "operator_supplied_unverified_history"
        ),
        "row_count": len(rows),
        "bar_seconds": bar_seconds,
        "source_first_ms": rows[0]["datetime"],
        "source_last_ms": rows[-1]["datetime"],
        "libraries": {name: version(name) for name in PACKAGES},
        "checks": checks,
        "analysis_gaps": (
            []
            if volatility["status"] == "estimated"
            else ["arch:" + volatility["status"]]
        )
        + [
            "pandas_ta_classic:" + name + ":undefined"
            for name in indicators["unavailable_indicators"]
        ],
        "analysis_warnings": (
            ["arch:persistence_at_or_above_one"]
            if volatility.get("stationary_fit") is False
            else []
        ),
        "additional_indicators": indicators,
        "additional_simulator": timing,
        "performance_diagnostics": performance,
        "volatility_diagnostics": volatility,
        "indicators": {
            "sma10": float(fast[-1]),
            "sma30": float(slow[-1]),
            "rsi14": float(talib.RSI(close, 14)[-1]),
            "atr14": float(talib.ATR(high, low, close, 14)[-1]),
        },
        "comparison": {
            "strategy": "fixed_sma10_30_example_not_native_bot",
            "fill_model": "next_open_one_share_long_only",
            "starting_cash": cash,
            "fee_rate": fee,
            "slippage_rate": slippage,
            "backtrader_final_equity": bt_value,
            "vectorbt_final_equity": vbt_value,
            "backtrader_fills": len(result.fills),
            "vectorbt_fills": len(vbt_fills),
            "rejected_orders": result.failures,
        },
        "storage": {
            "arrow_bytes": table.nbytes,
            "parquet_zstd_bytes": len(compressed),
            "rows": count,
            "min_close": min_close,
            "max_close": max_close,
        },
        "limitations": [
            "Example strategy, not a native bot recommendation or optimized strategy.",
            "No out-of-sample evaluation, corporate-action or dividend certification.",
            "Fixed costs are assumptions; no real liquidity, tax or fill guarantee.",
            "Open positions are marked at the final close, not forcibly liquidated.",
            "Input history completeness and provider provenance are not certified.",
            "Backtesting.py is a separate zero-cost timing control, not the cost-inclusive strategy result.",
            "Return metrics are per observed bar, not trade win rates or annualized performance.",
            "GARCH output is an unvalidated model estimate; missing or unconverged fits stay unavailable.",
        ],
        **AUTHORITY,
    }


def local_path(path, *, interpreter=False):
    from core.storage_router import inspect_storage_path

    path = Path(os.path.abspath(Path(path).expanduser()))
    route = inspect_storage_path(path, allow_external=False)
    allowed = {"present"} if interpreter else {"present", "missing"}
    if route.get("status") not in allowed or (
        route.get("symlinks") and not interpreter
    ):
        raise ValueError("ordinary_local_path_required")
    if interpreter and (route.get("kind") != "file" or not os.access(path, os.X_OK)):
        raise ValueError("executable_research_interpreter_required")
    # Preserve a venv executable path; resolving it would select its base environment.
    return path


def read_input(path):
    path = local_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("bounded_regular_input_required")
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input_byte_limit")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def run_worker(request, python):
    # Do not inherit credentials, runtime overrides, caches or broker routing.
    with tempfile.TemporaryDirectory(prefix="library-research-") as scratch:
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": scratch,
            "TMPDIR": scratch,
            "MPLCONFIGDIR": scratch,
            "NUMBA_CACHE_DIR": scratch,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NUMEXPR_MAX_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                [str(python), "-I", "-B", str(Path(__file__).resolve()), "--worker"],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.DEVNULL,
                cwd=scratch,
                env=env,
                start_new_session=True,
            )
            try:
                process.communicate(
                    json.dumps(request, allow_nan=False).encode(),
                    timeout=TIMEOUT_SECONDS,
                )
            except BaseException as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise ValueError("research_worker_timeout") from None
                raise
            output.seek(0)
            raw = output.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("research_output_byte_limit")
        try:
            report = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ValueError(
                f"research_worker_failed_exit_{process.returncode}"
            ) from None
        if (
            not isinstance(report, dict)
            or report.get("purpose") != "offline_library_comparison"
        ):
            raise ValueError("invalid_research_worker_report")
        if process.returncode:
            report["ok"] = False
        report.update(AUTHORITY)
        report["research_python"] = str(python)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--self-test", action="store_true")
    selection.add_argument(
        "--input",
        type=Path,
        help="Local Schwab-shaped candles JSON; epoch milliseconds",
    )
    selection.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bar-seconds", type=int)
    parser.add_argument(
        "--research-python",
        default=sys.executable,
        help="Offline interpreter only; defaults to the calling Python",
    )
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.worker:
            # Deny Python-level network access before importing research packages.
            def audit(event, _args):
                if event in {
                    "socket.connect",
                    "socket.getaddrinfo",
                    "socket.bind",
                    "socket.sendto",
                    "subprocess.Popen",
                    "os.system",
                }:
                    raise PermissionError("offline_research_only")

            sys.addaudithook(audit)
            request = json.loads(sys.stdin.buffer.read(MAX_BYTES + 1))
            report = compare(
                request["payload"],
                bar_seconds=request["bar_seconds"],
                synthetic=request["synthetic"],
            )
        else:
            if args.input and args.bar_seconds is None:
                raise ValueError("input_requires_explicit_bar_seconds")
            output_path = local_path(args.out_file) if args.out_file else None
            if output_path and args.input and output_path == local_path(args.input):
                raise ValueError("output_must_not_replace_input")
            payload, source_sha = (
                (fixture(), "synthetic") if args.self_test else read_input(args.input)
            )
            seconds = 3600 if args.self_test else args.bar_seconds
            validate_candles(payload, bar_seconds=seconds)
            report = run_worker(
                {
                    "payload": payload,
                    "bar_seconds": seconds,
                    "synthetic": args.self_test,
                },
                local_path(args.research_python, interpreter=True),
            )
            report["input_sha256"] = source_sha
            report["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if output_path:
                from scripts.ops.long_runtime_common import write_text_atomic

                write_text_atomic(
                    output_path, json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
        print(
            json.dumps(
                report, allow_nan=False, indent=None if args.json or args.worker else 2
            )
        )
        return 0 if report.get("ok") else 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "purpose": "offline_library_comparison",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    **AUTHORITY,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
