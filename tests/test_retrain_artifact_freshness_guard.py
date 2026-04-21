import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.retrain_artifact_freshness_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_check_distinguishes_sample_sufficiency_from_freshness(tmp_path: Path) -> None:
    artifact = tmp_path / "paper_replay_drill_latest.json"
    now = datetime.now(timezone.utc)
    _write_json(
        artifact,
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "failed_checks": ["paper_rows_low"],
        },
    )

    check = src._check(artifact, max_age_min=180.0, require_ok=True)

    assert check["ok"] is False
    assert check["failure_categories"] == ["sample_sufficiency"]
    assert check["failure_reasons"] == ["paper_rows_low"]


def test_main_reports_failure_category_buckets(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    replay = tmp_path / "paper_replay_drill_latest.json"
    recon = tmp_path / "paper_reconciliation_slo_latest.json"
    out_file = tmp_path / "retrain_artifact_freshness_latest.json"

    _write_json(
        replay,
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "failed_checks": ["paper_rows_low"],
        },
    )
    _write_json(
        recon,
        {
            "timestamp_utc": (now - timedelta(hours=5)).isoformat(),
            "ok": True,
            "failed_checks": [],
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retrain_artifact_freshness_guard.py",
            "--paper-replay-file",
            str(replay),
            "--paper-recon-file",
            str(recon),
            "--out-file",
            str(out_file),
            "--no-auto-refresh",
            "--no-auto-prune-stale",
            "--max-age-minutes",
            "180",
            "--json",
        ],
    )

    rc = src.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["sample_sufficiency_failed_checks"] == ["paper_replay"]
    assert payload["freshness_failed_checks"] == ["paper_reconciliation"]
    assert payload["failure_categories"]["sample_sufficiency"] == ["paper_replay"]
    assert payload["failure_categories"]["freshness"] == ["paper_reconciliation"]


def test_paper_replay_refresh_hours_plan_adds_fallback_window_without_duplicates() -> None:
    assert src._paper_replay_refresh_hours_plan(24, 72) == [24, 72]
    assert src._paper_replay_refresh_hours_plan(72, 72) == [72]
