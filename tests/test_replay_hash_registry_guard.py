import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.replay_hash_registry_guard as replay_guard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_replay_hash_registry_guard_auto_rebases_stale_expected_hash(tmp_path: Path, monkeypatch, capsys) -> None:
    now = datetime.now(timezone.utc)
    registry_path = tmp_path / "replay_expected_hashes.json"
    paper_path = tmp_path / "paper_replay_drill_latest.json"
    e2e_path = tmp_path / "replay_end_to_end_latest.json"
    out_path = tmp_path / "replay_hash_registry_guard_latest.json"

    _write_json(
        registry_path,
        {
            "paper_replay": {
                "all|all": {
                    "expected_hash": "old-paper-hash",
                    "updated_utc": (now - timedelta(days=30)).isoformat(),
                }
            },
            "replay_end_to_end": {},
        },
    )
    _write_json(
        paper_path,
        {
            "profile": "all",
            "domain": "all",
            "ok": True,
            "replay_hash": "new-paper-hash",
        },
    )
    _write_json(e2e_path, {"ok": True, "replay_hash": "fresh-e2e-hash"})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_hash_registry_guard.py",
            "--registry-file",
            str(registry_path),
            "--paper-file",
            str(paper_path),
            "--e2e-file",
            str(e2e_path),
            "--out-file",
            str(out_path),
            "--json",
        ],
    )

    rc = replay_guard.main()
    payload = json.loads(capsys.readouterr().out)
    updated_registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["ok"] is True
    assert payload["details"]["paper"]["auto_rebased"] is True
    assert updated_registry["paper_replay"]["all|all"]["expected_hash"] == "new-paper-hash"


def test_replay_hash_registry_guard_rebases_recent_expected_hash_when_source_hash_is_healthy(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    now = datetime.now(timezone.utc)
    registry_path = tmp_path / "replay_expected_hashes.json"
    paper_path = tmp_path / "paper_replay_drill_latest.json"
    e2e_path = tmp_path / "replay_end_to_end_latest.json"
    out_path = tmp_path / "replay_hash_registry_guard_latest.json"

    _write_json(
        registry_path,
        {
            "paper_replay": {
                "all|all": {
                    "expected_hash": "recent-old-paper-hash",
                    "updated_utc": (now - timedelta(hours=2)).isoformat(),
                    "auto_rebased": True,
                }
            },
            "replay_end_to_end": {},
        },
    )
    _write_json(
        paper_path,
        {
            "profile": "all",
            "domain": "all",
            "ok": True,
            "replay_hash": "fresh-paper-hash",
            "hash_match": True,
        },
    )
    _write_json(e2e_path, {"ok": True, "replay_hash": "fresh-e2e-hash"})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_hash_registry_guard.py",
            "--registry-file",
            str(registry_path),
            "--paper-file",
            str(paper_path),
            "--e2e-file",
            str(e2e_path),
            "--out-file",
            str(out_path),
            "--json",
        ],
    )

    rc = replay_guard.main()
    payload = json.loads(capsys.readouterr().out)
    updated_registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["ok"] is True
    assert payload["details"]["paper"]["hash_match"] is True
    assert updated_registry["paper_replay"]["all|all"]["expected_hash"] == "fresh-paper-hash"
