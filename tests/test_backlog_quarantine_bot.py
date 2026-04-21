import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import backlog_quarantine_bot as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_backlog_quarantine_bot_selects_only_previous_day_stale_candidates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    stale_file = project_root / "governance" / "shadow_crypto" / "shadow_pnl_attribution_20260406.jsonl"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("{\"pnl\": 1}\n", encoding="utf-8")
    old_mtime = (datetime.now(timezone.utc) - timedelta(hours=9)).timestamp()
    os.utime(stale_file, (old_mtime, old_mtime))

    fresh_file = project_root / "decision_explanations" / "shadow_crypto" / "decision_explanations_20260407.jsonl"
    fresh_file.parent.mkdir(parents=True, exist_ok=True)
    fresh_file.write_text("{\"why\": \"fresh\"}\n", encoding="utf-8")

    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "top_cold_pending_files": [
                {
                    "source_rel": "governance/shadow_crypto/shadow_pnl_attribution_20260406.jsonl",
                    "pending_lines": 180000,
                    "oldest_pending_age_seconds": 8 * 3600,
                }
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "decision_explanations/shadow_crypto/decision_explanations_20260407.jsonl",
                    "pending_lines": 60000,
                    "oldest_pending_age_seconds": 2 * 3600,
                }
            ],
        },
    )

    payload = src.build_payload(project_root, apply=False)

    assert payload["overall_status"] == "ready"
    assert payload["candidate_files"] == 1
    assert payload["candidate_rows"][0]["source_rel"].endswith("shadow_pnl_attribution_20260406.jsonl")


def test_backlog_quarantine_bot_applies_move_into_stale_stage(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    stale_root = project_root / "data" / "stale_stage"
    stale_manifest = stale_root / "backlog_quarantine_manifest.jsonl"
    stale_file = project_root / "governance" / "shadow_intraday_aggressive_equities" / "shadow_pnl_attribution_20260406.jsonl"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("{\"pnl\": 1}\n", encoding="utf-8")
    old_mtime = (datetime.now(timezone.utc) - timedelta(hours=10)).timestamp()
    os.utime(stale_file, (old_mtime, old_mtime))

    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "top_cold_pending_files": [
                {
                    "source_rel": "governance/shadow_intraday_aggressive_equities/shadow_pnl_attribution_20260406.jsonl",
                    "pending_lines": 220000,
                    "oldest_pending_age_seconds": 10 * 3600,
                }
            ],
            "top_deferred_pending_files": [],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        stale_stage_root=stale_root,
        stale_stage_manifest=stale_manifest,
    )

    assert payload["overall_status"] == "applied"
    assert payload["moved_files"] == 1
    assert payload["moved_pending_lines"] == 220000
    assert stale_manifest.exists()
    assert not stale_file.exists()


def test_backlog_quarantine_bot_dedupes_same_file_across_cold_and_deferred_lists(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    stale_file = project_root / "governance" / "shadow_crypto" / "shadow_pnl_attribution_20260406.jsonl"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("{\"pnl\": 1}\n", encoding="utf-8")
    old_mtime = (datetime.now(timezone.utc) - timedelta(hours=12)).timestamp()
    os.utime(stale_file, (old_mtime, old_mtime))

    row = {
        "source_rel": "governance/shadow_crypto/shadow_pnl_attribution_20260406.jsonl",
        "pending_lines": 300000,
        "oldest_pending_age_seconds": 11 * 3600,
    }
    _write_json(health / "ingestion_backpressure_latest.json", {"top_cold_pending_files": [row], "top_deferred_pending_files": [row]})

    payload = src.build_payload(project_root, apply=False)

    assert payload["candidate_files"] == 1
    assert payload["candidate_pending_lines"] == 300000
