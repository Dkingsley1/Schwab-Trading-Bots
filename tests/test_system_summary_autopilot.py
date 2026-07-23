from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.ops import system_summary_autopilot as src


def test_system_summary_autopilot_timeout_payload(monkeypatch) -> None:
    def _timeout(*_args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", ["fake"]), timeout=kwargs.get("timeout", 7))

    monkeypatch.setattr(src.subprocess, "run", _timeout)

    result = src._run(["fake-summary"], timeout_sec=7)

    assert result["rc"] == 124
    assert result["timeout_sec"] == 7
    assert "timeout_after_seconds=7" in result["stderr_tail"]


def test_system_summary_autopilot_passes_step_timeout_and_skips_bundle_when_suppressed(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    health_root.mkdir(parents=True)
    (health_root / "chrome_headless_guard_latest.json").write_text(
        '{"timeline_pdf_policy":"suppress"}',
        encoding="utf-8",
    )
    calls: list[tuple[list[str], int]] = []

    def _fake_run(cmd: list[str], timeout_sec: int = src.DEFAULT_STEP_TIMEOUT_SEC) -> dict:
        calls.append((list(cmd), timeout_sec))
        return {
            "rc": 0,
            "payload": {
                "overall_status": "ready",
                "pdf": {"enabled": False, "ok": False},
                "html_paths": {"latest": "summary.html"},
                "section_grade_board": {"overall_letter_grade": "A+"},
            },
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_ms": 1.0,
            "timeout_sec": timeout_sec,
        }

    monkeypatch.setattr(src, "_run", _fake_run)

    payload = src.build_payload(tmp_path, step_timeout_sec=11)

    assert payload["overall_status"] == "ready"
    assert payload["step_timeout_sec"] == 11
    assert len(calls) == 1
    assert calls[0][1] == 11
    assert payload["report_bundle"]["payload_summary"]["overall_status"] == "skipped"


def test_system_summary_autopilot_env_quiet_mode_overrides_stale_allow_artifact(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    health_root.mkdir(parents=True)
    (health_root / "chrome_headless_guard_latest.json").write_text(
        '{"timeline_pdf_policy":"allow"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("REPORT_HEADLESS_BROWSER_RENDER_ENABLED", "0")
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], timeout_sec: int = src.DEFAULT_STEP_TIMEOUT_SEC) -> dict:
        calls.append(list(cmd))
        return {
            "rc": 0,
            "payload": {
                "overall_status": "ready",
                "pdf": {"enabled": False, "ok": False},
                "html_paths": {"latest": "summary.html"},
                "section_grade_board": {"overall_letter_grade": "A+"},
            },
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_ms": 1.0,
            "timeout_sec": timeout_sec,
        }

    monkeypatch.setattr(src, "_run", _fake_run)

    payload = src.build_payload(tmp_path, step_timeout_sec=11)

    assert payload["chrome_policy"] == "suppress"
    assert payload["quiet_mode_active"] is True
    assert payload["refresh_supporting_artifacts"] is False
    assert len(calls) == 1
    assert "--refresh-supporting-artifacts" not in calls[0]
    assert "--no-render-pdf" in calls[0]
    assert "--no-allow-gui-pdf-renderer" in calls[0]
