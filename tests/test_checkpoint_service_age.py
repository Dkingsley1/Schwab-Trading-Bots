from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts import ingestion_backpressure_guard as guard


def observation():
    return SimpleNamespace(st_ino=10, st_size=200, st_mtime=9999), dict(
        last_line=1,
        last_offset_bytes=100,
        file_size_bytes=100,
        file_inode=10,
        mtime=2000,
    )


def test_checkpoint_age_is_separate_from_source_age_and_backlog_count():
    stat, progress = observation()
    age = guard._checkpoint_service_age_seconds(stat, progress, 10000)
    assert age == 8000
    rows = []
    guard._record_top_pending(
        rows,
        rel="governance/channels/api/test.jsonl",
        pending=2,
        age_seconds=1,
        total=3,
        last_line=1,
        top_n=10,
        line_estimate={"checkpoint_service_age_seconds": age},
    )
    assert rows[0]["checkpoint_service_age_seconds"] == 8000
    assert rows[0]["oldest_pending_age_seconds"] == 1
    assert rows[0]["pending_lines"] == 2


@pytest.mark.parametrize(
    "change",
    [
        {"file_inode": 11},
        {"file_inode": 0},
        {"last_line": 0},
        {"last_offset_bytes": 0},
        {"last_offset_bytes": 200},
        {"last_offset_bytes": 201},
        {"file_size_bytes": 300},
        {"mtime": 10001},
        {"mtime": float("nan")},
        {"mtime": float("inf")},
        {"mtime": 0},
        {"mtime": "bad"},
    ],
)
def test_invalid_reset_or_eof_checkpoint_cannot_supply_service_age(change):
    stat, progress = observation()
    assert (
        guard._checkpoint_service_age_seconds(stat, {**progress, **change}, 10000) == 0
    )
