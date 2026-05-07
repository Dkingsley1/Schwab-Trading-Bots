import gzip
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import bot_logs_cleanup_intelligence as cleanup


def _fake_disk(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": True,
        "total_bytes": 10 * 1024**3,
        "used_bytes": 9 * 1024**3,
        "free_bytes": 1,
        "free_gb": 0.0,
        "used_gb": 9.0,
        "capacity_pct": 99.0,
    }


def _write_gzip_pair(raw_path: Path, content: bytes, *, age_hours: float = 48.0) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    with gzip.open(str(raw_path) + ".gz", "wb") as handle:
        handle.write(content)
    old_ts = time.time() - (age_hours * 3600.0)
    os.utime(raw_path, (old_ts, old_ts))
    os.utime(str(raw_path) + ".gz", (old_ts, old_ts))


def test_cleanup_selects_old_verified_gzip_duplicates(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    _write_gzip_pair(root / "governance" / "execution_lanes" / "execution_results_20260501.jsonl", b'{"ok": true}\n' * 200)
    (root / "decisions" / "paper").mkdir(parents=True)
    (root / "decisions" / "paper" / "trade_decisions_20260501.jsonl").write_text("raw only\n", encoding="utf-8")

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        target_free_gb=1.0,
        max_tier=1,
        min_age_hours=12.0,
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["candidate_summary"]["eligible_count"] == 1
    assert payload["selected_count"] == 1
    assert payload["selected_candidates"][0]["relative_path"] == "governance/execution_lanes/execution_results_20260501.jsonl"


def test_cleanup_apply_deletes_raw_and_keeps_gzip(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    raw_path = root / "governance" / "execution_lanes" / "execution_results_20260501.jsonl"
    _write_gzip_pair(raw_path, b'{"row": 1}\n' * 300)

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        apply=True,
        target_free_gb=1.0,
        max_tier=1,
        min_age_hours=12.0,
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["apply_result"]["deleted_files"] == 1
    assert not raw_path.exists()
    assert Path(str(raw_path) + ".gz").exists()
    assert (tmp_path / "history.jsonl").read_text(encoding="utf-8").strip()


def test_cleanup_protects_current_day_duplicates(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    raw_path = root / "decisions" / "paper" / "trade_decisions_20990101.jsonl"
    _write_gzip_pair(raw_path, b'{"active": true}\n' * 100, age_hours=72.0)
    monkeypatch.setattr(cleanup, "_today_tokens", lambda now=None: {"20990101"})

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        target_free_gb=1.0,
        max_tier=1,
        min_age_hours=12.0,
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["candidate_summary"]["eligible_count"] == 0
    assert payload["selected_count"] == 0
    assert payload["top_candidates"][0]["blocked_reasons"] == ["current_day_protected"]


def test_cleanup_offloads_external_local_fallback_conflicts(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    conflict_path = (
        root
        / "decision_explanations"
        / "shadow_crypto"
        / "decision_explanations_20260507.jsonl.local_fallback.2"
    )
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    conflict_path.write_text("conflict-copy\n", encoding="utf-8")
    quarantine_root = tmp_path / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup"

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        apply=True,
        target_free_gb=1.0,
        max_tier=2,
        fallback_quarantine_root=quarantine_root,
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    quarantined = quarantine_root / "decision_explanations" / "shadow_crypto" / conflict_path.name
    assert payload["apply_result"]["offloaded_files"] == 1
    assert payload["apply_result"]["deleted_files"] == 0
    assert not conflict_path.exists()
    assert quarantined.read_text(encoding="utf-8") == "conflict-copy\n"
