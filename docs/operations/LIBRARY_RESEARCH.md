# Offline Library Research

## Implemented Uses

- `scripts/jsonl_codec.py` uses optional Msgspec JSON decoding in the bounded market-cycle and paper-profitability JSONL readers. Standard-library fallback preserves legacy nonfinite values, escaped surrogates and backend failures. Existing byte/row bounds, timestamps and report semantics are unchanged.
- `scripts/ops/library_research.py` uses Polars rolling transforms, TA-Lib SMA/RSI/ATR, Backtrader event simulation and VectorBT vectorized simulation. It compares a fixed SMA10/30 example using one-share long-only, next-open fills, 10 bps fees and 3 bps modeled slippage.
- `scripts/ops/library_research_extensions.py` adds Pandas TA Classic native SMA/RSI/ATR (`talib=False`), explicit SMA agreement and prefix-invariance checks, and a Backtesting.py timing control. The latter runs with zero costs and compares next-open bar, side, price, fill count and final equity against a separate zero-cost VectorBT reference. It must not be confused with the original cost-inclusive result; no optimizer or plotting server runs.
- QuantStats reports compounded return, maximum drawdown, per-observed-bar volatility and positive-bar fraction from the cost-inclusive simulated equity. A fixed starting-equity baseline is included and return/drawdown must reconcile independently. Metrics are not annualized, and positive bars are not trade wins.
- `arch` fits one zero-mean GARCH(1,1) model to the last 1,000 observed percentage returns at most, requiring 100 variable returns and at most 100 optimizer iterations. It reports one-step volatility in fractional-return units, not an annualized or validated forecast. Short/flat histories retain null results and `analysis_gaps`; failed convergence or invalid forecasts fail the check. Persistence at or above one remains an explicit `analysis_warnings` entry. An `ok` operational comparison can contain unavailable analysis checks; it never certifies model usefulness.
- The research command verifies an in-memory Zstandard Parquet round trip with PyArrow and performs a bounded in-memory DuckDB aggregate. Existing production DuckDB/Parquet owners remain unchanged. Small files can be larger after Parquet encoding; no storage savings are promised by this check.
- Ray, Redis and Hiredis are not required. Redis was removed from the platform dependency lock, live profile and runtime routing. Do not install VectorBT's optional `full`/`all` extras, which can pull Ray back in.

## Runtime Separation

The downloaded research packages were found in the framework Python 3.14 installation, not the platform's `.venv314`. Reuse that installation explicitly; do not add its site-packages to the live Python path or silently upgrade the live environment. The report records the actual interpreter and all eleven library versions. On another host, provide its reviewed research interpreter. Without the flag, the command uses the calling interpreter and fails clearly if packages are missing.

`config/library_research_extras.lock.txt` pins the four added packages and the two previously missing Backtesting.py dependencies (Bokeh and xyzservices). It is an additions-only lock, not a complete frozen environment. The reviewed install added approximately 38 MiB without upgrading existing packages. QuantStats and arch remain separately installed in the platform runtime as before. Backtesting.py's installed metadata declares `AGPL-3.0`; review redistribution and licensing before incorporating this integration into a distributed product. No commercial-readiness clearance is granted by installation.

```sh
./scripts/ops/opsctl.sh library-research --research-python /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 --self-test --json
```

To inspect local Schwab-shaped candles, replace `--self-test` with `--input /absolute/local/candles.json --bar-seconds 3600`, using the actual candle duration. Input must be an object containing a `candles` list with `datetime` (Unix milliseconds), `open`, `high`, `low`, `close`, and `volume`. All bars must be ordered, unique, finite and closed. The command rejects invalid input rather than filling gaps, sorting duplicate bars or inventing prices. It does not certify session coverage, adjustments or provider provenance.

Use `--out-file /absolute/local/report.json` to retain a compact report. The input cannot be the output. No report or raw-data duplicate is written by default; intermediate Parquet bytes stay in memory and child caches are temporary.

## Bounds And Authority

Input is capped at 4 MiB and 5,000 candles, with a minimum of 40. One child has a 90-second wall deadline, one-thread library settings and no inherited broker credentials or runtime overlays. Python network/subprocess audit events are denied before research imports. This is a defense-in-depth restriction on trusted research libraries, not an OS security sandbox for arbitrary code. DuckDB uses one thread, 128 MiB and disabled external access. Symlinked and external inputs/outputs, including protected volumes, are rejected. Timeout kills and reaps the child process group.

The fixed example is not a native bot decision. It does not select strategies, optimize parameters, contact a broker, trade, update runtime signals, clear halts, accept source, grant promotion credit or establish profitability. Open simulated positions are marked to the final close, not liquidated. Dividend/corporate-action, holdout, realistic liquidity and tax work remain separate. No automatic schedule or live service was added.

## Verification

```sh
LIBRARY_RESEARCH_TEST_PYTHON=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 .venv314/bin/python -m pytest -q tests/test_jsonl_codec.py tests/test_library_research.py tests/test_library_research_extensions.py
```

The optional real-engine test requires that explicit interpreter; remaining validation, fallback, input-boundary and timeout tests run in the platform environment. Source edits still require the platform's normal reviewed release process before acceptance. Installation, a passing fixture and source acceptance are distinct states.

Primary API references: [TA-Lib](https://github.com/TA-Lib/ta-lib-python), [Backtrader Cerebro](https://www.backtrader.com/docu/cerebro/), [VectorBT portfolio](https://vectorbt.dev/api/portfolio/base/).

Extension references: [Pandas TA Classic](https://xgboosted.github.io/pandas-ta-classic/indicators.html), [Backtesting.py](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html), [QuantStats](https://github.com/ranaroussi/quantstats), [arch](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_forecasting.html).
