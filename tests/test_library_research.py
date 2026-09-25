from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.ops import library_research as src


def test_fixture_is_bounded_and_closed():
    rows = src.validate_candles(src.fixture(), bar_seconds=3600)
    assert len(rows) == 180


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda p: p["candles"].__setitem__(1, p["candles"][0]), "ordered_unique"),
        (lambda p: p["candles"][0].update(close=float("nan")), "invalid_candle_number"),
        (lambda p: p["candles"][0].update(volume=-1), "invalid_ohlcv"),
        (lambda p: p["candles"][0].update(high=1), "invalid_ohlcv"),
        (lambda p: p["candles"][0].update(open=True), "invalid_candle_number"),
        (lambda p: p.update(candles=p["candles"][:39]), "40_to_5000"),
        (lambda p: p.update(candles=p["candles"] * 30), "40_to_5000"),
    ],
)
def test_bad_inputs_rejected(change, reason):
    payload = src.fixture()
    change(payload)
    with pytest.raises(ValueError, match=reason):
        src.validate_candles(payload, bar_seconds=3600)


def test_unclosed_bars_rejected():
    payload = src.fixture()
    now = payload["candles"][-1]["datetime"] / 1000 + 3599
    with pytest.raises(ValueError, match="unclosed"):
        src.validate_candles(payload, bar_seconds=3600, now=now)


def test_input_has_byte_limit_and_digest(tmp_path):
    path = tmp_path / "candles.json"
    path.write_text(json.dumps(src.fixture()))
    payload, digest = src.read_input(path)
    assert payload == src.fixture()
    assert len(digest) == 64
    path.write_bytes(b" " * (src.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="bounded_regular"):
        src.read_input(path)


def test_protected_and_symlink_paths_rejected(tmp_path):
    with pytest.raises(ValueError, match="ordinary_local"):
        src.local_path("/Volumes/VIDEO/must-not-inspect.json")
    link = tmp_path / "link"
    link.symlink_to("/Volumes/VIDEO")
    with pytest.raises(ValueError, match="ordinary_local"):
        src.read_input(link / "must-not-inspect.json")


def test_main_requires_bar_duration_and_preserves_input(tmp_path, capsys):
    path = tmp_path / "candles.json"
    path.write_text(json.dumps(src.fixture()))
    before = path.read_bytes()
    assert src.main(["--input", str(path)]) == 2
    assert "explicit_bar_seconds" in capsys.readouterr().out
    assert (
        src.main(
            ["--input", str(path), "--bar-seconds", "3600", "--out-file", str(path)]
        )
        == 2
    )
    assert path.read_bytes() == before


def test_interpreter_symlink_does_not_escape_or_lose_venv_identity(tmp_path):
    path = tmp_path / "python"
    path.symlink_to(src.sys.executable)
    assert src.local_path(path, interpreter=True) == path
    with pytest.raises(ValueError, match="ordinary_local"):
        src.local_path("/Volumes/VIDEO/python", interpreter=True)


def test_timeout_kills_child_and_does_not_pass_credentials(monkeypatch):
    observed = {}

    class Child:
        pid = 99999

        def communicate(self, data, timeout):
            assert timeout == src.TIMEOUT_SECONDS
            assert json.loads(data) == {"fixture": True}
            raise subprocess.TimeoutExpired("worker", timeout)

        def wait(self):
            observed["waited"] = True

    def spawn(command, **kwargs):
        observed.update(kwargs)
        observed["command"] = command
        return Child()

    monkeypatch.setenv("SCHWAB_APP_SECRET", "not_for_child")
    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: observed.update(killed=pid))
    with pytest.raises(ValueError, match="worker_timeout"):
        src.run_worker({"fixture": True}, "/some/python")
    assert "SCHWAB_APP_SECRET" not in observed["env"]
    assert observed["env"]["POLARS_MAX_THREADS"] == "1"
    assert observed["env"]["NUMBA_NUM_THREADS"] == "1"
    assert observed["start_new_session"] is True
    assert observed["killed"] == 99999 and observed["waited"]
    assert not Path(observed["cwd"]).exists()


def test_real_engines_optional():
    python = os.environ.get("LIBRARY_RESEARCH_TEST_PYTHON")
    if not python:
        pytest.skip(
            "explicit research interpreter required for native package integration"
        )
    report = src.run_worker(
        {"payload": src.fixture(), "bar_seconds": 3600, "synthetic": True}, python
    )
    assert report["ok"], report
    assert all(report["checks"].values())
    assert report["comparison"]["backtrader_fills"] > 0
    assert report["data_kind"] == "synthetic_fixture"
    assert all(report[k] is False for k in src.AUTHORITY)
    assert set(report["libraries"]) == set(src.PACKAGES)
    assert report["additional_simulator"]["costs_included"] is False
    assert report["performance_diagnostics"]["annualized"] is False
    assert report["volatility_diagnostics"]["out_of_sample_validated"] is False


@pytest.mark.parametrize(
    "case,gap", [("short", "insufficient_history"), ("flat", "insufficient_variation")]
)
def test_real_worker_retains_unavailable_volatility(case, gap):
    python = os.environ.get("LIBRARY_RESEARCH_TEST_PYTHON")
    if not python:
        pytest.skip("explicit research interpreter required")
    payload = src.fixture()
    if case == "short":
        payload["candles"] = payload["candles"][:40]
    else:
        for row in payload["candles"]:
            row.update(open=50, high=51, low=49, close=50)
    report = src.run_worker(
        {"payload": payload, "bar_seconds": 3600, "synthetic": True}, python
    )
    assert report["ok"], report
    assert report["checks"]["arch_fit_converged"] is None
    assert "arch:" + gap in report["analysis_gaps"]
    if case == "flat":
        assert report["additional_indicators"]["rsi14"] is None
        assert "pandas_ta_classic:rsi14:undefined" in report["analysis_gaps"]
    assert report["volatility_diagnostics"]["next_bar_volatility"] is None
    json.dumps(report, allow_nan=False)


def test_catalog_and_removed_dependencies():
    root = src.ROOT
    candidates = json.loads(
        (root / "config/library_candidate_routes_v1.json").read_text()
    )
    names = {row["package"] for row in candidates["candidate_libraries"]}
    assert {
        "ta-lib",
        "vectorbt",
        "backtrader",
        "pandas-ta-classic",
        "backtesting",
        "quantstats",
        "arch",
    } <= names
    from scripts.ops.dependency_activation_smoke import _module_name
    from scripts.ops.runtime_dependency_profiles import PROFILE_RULES

    assert _module_name("TA-Lib") == "talib"
    assert _module_name("pandas-ta-classic") == "pandas_ta_classic"
    activation = json.loads(
        (root / "config/library_activation_profiles_v1.json").read_text()
    )
    for package in ("pandas-ta-classic", "backtesting", "quantstats", "arch"):
        assert activation["package_profile_overrides"][package] == ["research"]
    additions = (
        (root / "config/library_research_extras.lock.txt").read_text().splitlines()
    )
    pins = [line for line in additions if line and not line.startswith("#")]
    assert len(pins) == 6 and all("==" in line and "[" not in line for line in pins)
    assert not any(line.startswith(("ray", "redis", "hiredis")) for line in pins)
    assert "redis" not in set().union(*PROFILE_RULES.values())
    for relative in [
        "config/requirements.lock.txt",
        "config/runtime_profiles/live.lock.txt",
    ]:
        lines = (root / relative).read_text().lower().splitlines()
        assert not any(
            row.startswith(("redis==", "ray==", "hiredis==")) for row in lines
        )
