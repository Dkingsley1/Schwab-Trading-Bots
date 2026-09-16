import signal
from pathlib import Path

from scripts.ops import mlx_audio_runtime_audit as src


def test_package_rows_flag_missing_audio_runtime_packages() -> None:
    rows, ok = src._package_rows(
        ("mlx-audio", "mlx", "mlx-lm", "transformers", "miniaudio"),
        {
            "mlx-audio": "0.4.0",
            "mlx": "0.31.1",
            "transformers": "5.0.0rc3",
        },
    )

    assert ok is False
    assert rows == [
        {"package": "mlx-audio", "installed_version": "0.4.0", "status": "ok"},
        {"package": "mlx", "installed_version": "0.31.1", "status": "ok"},
        {"package": "mlx-lm", "installed_version": None, "status": "missing_runtime"},
        {"package": "transformers", "installed_version": "5.0.0rc3", "status": "ok"},
        {
            "package": "miniaudio",
            "installed_version": None,
            "status": "missing_runtime",
        },
    ]


def test_audio_probe_sequence_stops_after_native_abort(monkeypatch) -> None:
    calls: list[str] = []

    def fake_step(name, _cmd, accepted_rc=None):
        calls.append(name)
        return {
            "name": name,
            "ok": False,
            "failure_kind": "native_signal",
            "termination_signal": "SIGABRT",
        }

    monkeypatch.setattr(src, "_step", fake_step)
    results = src._run_probe_sequence(
        [
            ("mlx_core_import", ["python", "-c", "pass"]),
            ("mlx_audio_import", ["python", "-c", "pass"]),
        ],
        prerequisite_ok=True,
    )

    assert calls == ["mlx_core_import"]
    assert results[0]["termination_signal"] == "SIGABRT"
    assert results[1]["failure_kind"] == "prerequisite_blocked"
    assert results[1]["blocked_by"] == "mlx_core_import"


def test_audio_step_classifies_native_abort(monkeypatch) -> None:
    monkeypatch.setattr(
        src,
        "_run",
        lambda _cmd: (-signal.SIGABRT, "", "nanobind: critical error"),
    )

    result = src._step("mlx_audio_import", ["python", "-c", "pass"])

    assert result["ok"] is False
    assert result["failure_kind"] == "native_signal"
    assert result["termination_signal"] == "SIGABRT"


def test_inventory_step_classifies_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        src,
        "_run",
        lambda _cmd: (124, "", "process_timeout_after=60s"),
    )

    versions, step = src._load_installed_versions(Path("/tmp/python"))

    assert versions == {}
    assert step["ok"] is False
    assert step["failure_kind"] == "timeout"
