import signal
from pathlib import Path

from scripts.ops import mlx_runtime_audit as src


def test_default_packages_cover_mlx_research_extensions() -> None:
    assert {
        "mlx-graphs",
        "mlx-snn",
        "mlx-vision",
        "esig",
        "roughpy",
    }.issubset(src.DEFAULT_PACKAGES)


def test_package_rows_detect_drift_states() -> None:
    rows, ok = src._package_rows(
        (
            "mlx",
            "mlx-data",
            "mlx-vlm",
            "mlx-whisper",
            "transformers",
            "schwab-py",
            "duckdb",
        ),
        {
            "mlx": "0.30.6",
            "mlx-data": "0.2.0",
            "mlx-vlm": "0.4.4",
            "transformers": "5.3.0",
            "schwab-py": "1.5.1",
        },
        {
            "mlx": "0.31.0",
            "mlx-data": "0.2.0",
            "mlx-vlm": "0.4.4",
            "mlx-whisper": "0.4.3",
            "transformers": "5.3.0",
            "duckdb": "1.5.0",
        },
    )

    assert ok is False
    assert rows == [
        {
            "package": "mlx",
            "locked_version": "0.30.6",
            "installed_version": "0.31.0",
            "status": "runtime_ahead_of_lock",
        },
        {
            "package": "mlx-data",
            "locked_version": "0.2.0",
            "installed_version": "0.2.0",
            "status": "ok",
        },
        {
            "package": "mlx-vlm",
            "locked_version": "0.4.4",
            "installed_version": "0.4.4",
            "status": "ok",
        },
        {
            "package": "mlx-whisper",
            "locked_version": None,
            "installed_version": "0.4.3",
            "status": "missing_lock",
        },
        {
            "package": "transformers",
            "locked_version": "5.3.0",
            "installed_version": "5.3.0",
            "status": "ok",
        },
        {
            "package": "schwab-py",
            "locked_version": "1.5.1",
            "installed_version": None,
            "status": "missing_runtime",
        },
        {
            "package": "duckdb",
            "locked_version": None,
            "installed_version": "1.5.0",
            "status": "missing_lock",
        },
    ]


def test_recommendations_highlight_direct_stable_compile_when_safe() -> None:
    recommendations = src._recommendations(
        [
            {
                "package": "mlx",
                "locked_version": "0.31.0",
                "installed_version": "0.31.0",
                "status": "ok",
            }
        ],
        {
            "compile_available": True,
            "compile_smoke_ok": True,
            "metal_available": True,
            "jit_env": "0",
        },
    )

    assert "candidate_mlx_compile_direct_stable_after_smoke" in recommendations
    assert "mlx_metal_jit_default_off" in recommendations


def test_recommendations_hold_compile_rollout_on_failed_smoke() -> None:
    recommendations = src._recommendations(
        [],
        {
            "compile_available": True,
            "compile_smoke_ok": False,
            "metal_available": True,
            "jit_env": "0",
        },
    )

    assert recommendations == [
        "keep_mlx_compile_opt_in_until_compile_smoke_passes",
        "mlx_metal_jit_default_off",
    ]


def test_step_classifies_native_abort_as_hard_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        src,
        "_run",
        lambda _cmd: (-signal.SIGABRT, "", "nanobind: critical error"),
    )

    result = src._step("native_import", ["python", "-I", "-c", "pass"])

    assert result["ok"] is False
    assert result["failure_kind"] == "native_signal"
    assert result["termination_signal"] == "SIGABRT"
    assert result["termination_signal_number"] == signal.SIGABRT


def test_step_classifies_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        src,
        "_run",
        lambda _cmd: (124, "", "process_timeout_after=60s"),
    )

    result = src._step("slow_import", ["python", "-I", "-c", "pass"])

    assert result["ok"] is False
    assert result["failure_kind"] == "timeout"


def test_probe_sequence_stops_after_native_abort(monkeypatch) -> None:
    calls: list[str] = []

    def fake_step(name, _cmd, accepted_rc=None):
        calls.append(name)
        if name == "first":
            return {
                "name": name,
                "ok": False,
                "failure_kind": "native_signal",
            }
        return {"name": name, "ok": True, "failure_kind": "none"}

    monkeypatch.setattr(src, "_step", fake_step)
    results = src._run_probe_sequence(
        [
            ("first", ["python", "-c", "pass"], None, ""),
            ("second", ["python", "-c", "pass"], None, ""),
            ("third", ["python", "-c", "pass"], None, ""),
        ],
        prerequisite_ok=True,
        prerequisite_name="runtime",
    )

    assert calls == ["first"]
    assert [row["failure_kind"] for row in results] == [
        "native_signal",
        "prerequisite_blocked",
        "prerequisite_blocked",
    ]
    assert results[1]["blocked_by"] == "first"


def test_probe_sequence_does_not_run_after_failed_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        src,
        "_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native probe should not run")
        ),
    )

    results = src._run_probe_sequence(
        [("mlx_core_import", ["python", "-c", "pass"], None, "")],
        prerequisite_ok=False,
        prerequisite_name="mlx_runtime_snapshot",
    )

    assert results == [
        {
            "name": "mlx_core_import",
            "ok": False,
            "rc": 125,
            "command": "blocked",
            "accepted_rc": [0],
            "stdout_tail": "",
            "stderr_tail": "native_probe_circuit_open:mlx_runtime_snapshot",
            "failure_kind": "prerequisite_blocked",
            "blocked_by": "mlx_runtime_snapshot",
        }
    ]


def test_indicator_import_uses_isolated_canonical_package_route() -> None:
    command = src._indicator_import_command(Path("/venv/bin/python"))

    assert command[:3] == ["/venv/bin/python", "-I", "-c"]
    assert "import core.indicator_bot_common as mod" in command[3]
    assert "sys.path.insert(0, 'core')" not in command[3]
    assert "import indicator_bot_common as mod" not in command[3]


def test_indicator_common_uses_canonical_runtime_training_import() -> None:
    source = (src.PROJECT_ROOT / "core" / "indicator_bot_common.py").read_text(
        encoding="utf-8"
    )

    assert "from core.runtime_training_common import (" in source
    assert "\nfrom runtime_training_common import (" not in source
