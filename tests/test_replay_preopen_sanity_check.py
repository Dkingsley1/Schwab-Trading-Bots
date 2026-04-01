import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import replay_preopen_sanity_check as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_summary_metrics_uses_daily_runtime_summary_when_available(tmp_path: Path) -> None:
    original_root = src.PROJECT_ROOT
    try:
        src.PROJECT_ROOT = tmp_path
        now = datetime(2026, 3, 26, 18, 0, tzinfo=timezone.utc)
        summary = tmp_path / "exports" / "sql_reports" / "daily_runtime_summary_20260326.json"
        _write_json(
            summary,
            {
                "day": "20260326",
                "decision": {"rows": 1234, "stale_windows": 1, "files": ["a", "b"]},
                "governance": {"rows": 456, "stale_windows": 2, "files": ["c"]},
            },
        )

        metrics = src._summary_metrics(now, profile="", domain="")

        assert metrics is not None
        assert metrics["source"] == "daily_runtime_summary"
        assert metrics["decision"]["rows"] == 1234
        assert metrics["decision"]["files"] == 2
        assert metrics["governance"]["rows"] == 456
        assert metrics["governance"]["stale_windows"] == 2
    finally:
        src.PROJECT_ROOT = original_root


def test_summary_metrics_skips_filtered_profile_requests(tmp_path: Path) -> None:
    original_root = src.PROJECT_ROOT
    try:
        src.PROJECT_ROOT = tmp_path
        now = datetime(2026, 3, 26, 18, 0, tzinfo=timezone.utc)
        _write_json(
            tmp_path / "exports" / "sql_reports" / "daily_runtime_summary_20260326.json",
            {
                "day": "20260326",
                "decision": {"rows": 1234, "stale_windows": 1, "files": ["a", "b"]},
                "governance": {"rows": 456, "stale_windows": 2, "files": ["c"]},
            },
        )

        assert src._summary_metrics(now, profile="fx", domain="") is None
    finally:
        src.PROJECT_ROOT = original_root


def test_summary_metrics_prefers_previous_day_when_current_day_is_empty(tmp_path: Path) -> None:
    original_root = src.PROJECT_ROOT
    try:
        src.PROJECT_ROOT = tmp_path
        now = datetime(2026, 3, 27, 0, 36, tzinfo=timezone.utc)
        _write_json(
            tmp_path / "exports" / "sql_reports" / "daily_runtime_summary_20260327.json",
            {
                "day": "20260327",
                "decision": {"rows": 0, "stale_windows": 0, "files": []},
                "governance": {"rows": 0, "stale_windows": 0, "files": []},
            },
        )
        previous = tmp_path / "exports" / "sql_reports" / "daily_runtime_summary_20260326.json"
        _write_json(
            previous,
            {
                "day": "20260326",
                "decision": {"rows": 1234, "stale_windows": 1, "files": ["a", "b"]},
                "governance": {"rows": 456, "stale_windows": 2, "files": ["c"]},
            },
        )

        metrics = src._summary_metrics(now, profile="", domain="")

        assert metrics is not None
        assert metrics["path"] == str(previous)
        assert metrics["decision"]["rows"] == 1234
    finally:
        src.PROJECT_ROOT = original_root
