from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.schwab_account_snapshot_refresh import _publish_refresh_summary


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_successful_refresh_publishes_canonical_and_last_good(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.json"
    attempt = tmp_path / "attempt.json"
    last_good = tmp_path / "last-good.json"

    result = _publish_refresh_summary(
        {"ok": True, "timestamp_utc": "2026-08-28T12:00:00+00:00"},
        project_root=tmp_path,
        out_path=canonical,
        attempt_path=attempt,
        last_good_path=last_good,
    )

    assert result["published_as_canonical"] is True
    assert _read(canonical)["ok"] is True
    assert _read(last_good)["ok"] is True
    assert _read(attempt)["ok"] is True


def test_failed_refresh_preserves_verified_canonical_truth(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.json"
    attempt = tmp_path / "attempt.json"
    last_good = tmp_path / "last-good.json"
    good = {"ok": True, "timestamp_utc": "2026-08-28T12:00:00+00:00"}
    _publish_refresh_summary(
        good,
        project_root=tmp_path,
        out_path=canonical,
        attempt_path=attempt,
        last_good_path=last_good,
    )

    failed = _publish_refresh_summary(
        {
            "ok": False,
            "timestamp_utc": "2026-08-28T12:05:00+00:00",
            "error": "provider_unavailable",
        },
        project_root=tmp_path,
        out_path=canonical,
        attempt_path=attempt,
        last_good_path=last_good,
    )

    assert failed["previous_good_preserved"] is True
    assert failed["published_as_canonical"] is False
    assert _read(canonical) == _read(last_good)
    assert _read(canonical)["timestamp_utc"] == good["timestamp_utc"]
    assert _read(attempt)["error"] == "provider_unavailable"
