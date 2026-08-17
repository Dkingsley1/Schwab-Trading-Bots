import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.execution_queue_stress_bot as queue_stress


def test_execution_queue_stress_uses_bounded_tail_window(tmp_path: Path, monkeypatch, capsys) -> None:
    shadow_dir = tmp_path / "governance" / "shadow_test"
    shadow_dir.mkdir(parents=True)
    path = shadow_dir / "master_control_test.jsonl"
    old_row = {
        "timestamp_utc": "2020-01-01T00:00:00+00:00",
        "portfolio": {"queue_depth": 999999},
    }
    recent_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "portfolio": {"queue_depth": 2500},
    }
    path.write_text(
        json.dumps(old_row) + "\n" + (" " * 4096) + "\n" + json.dumps(recent_row) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(queue_stress, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execution_queue_stress_bot.py",
            "--tail-bytes",
            "2048",
            "--max-queue-depth",
            "2000",
            "--json",
        ],
    )

    rc = queue_stress.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert payload["samples"] == 1
    assert payload["queue_depth_breaches"] == 1
    assert payload["scan_policy"] == "mtime_prefilter_then_tail_window"
