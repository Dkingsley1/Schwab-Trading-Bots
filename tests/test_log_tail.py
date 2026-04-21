import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.log_tail import count_tail_keyword, iter_tail_lines_reverse


def test_iter_tail_lines_reverse_handles_no_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "tail.log"
    path.write_text("first\nsecond\nthird", encoding="utf-8")

    lines = list(iter_tail_lines_reverse(path, max_bytes=1024, chunk_bytes=4))

    assert lines[:3] == ["third", "second", "first"]


def test_count_tail_keyword_handles_utf8_split_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "utf8.log"
    path.write_text("steady\npi \u03c0 restart one\nsteady\npi \u03c0 restart two\n", encoding="utf-8")

    count = count_tail_keyword(path, "restart", tail_lines=4, max_bytes=256, chunk_bytes=5)

    assert count == 2


def test_count_tail_keyword_limits_to_requested_tail(tmp_path: Path) -> None:
    path = tmp_path / "tail_only.log"
    path.write_text("restart old\nsteady\nrestart recent\nsteady recent\n", encoding="utf-8")

    count = count_tail_keyword(path, "restart", tail_lines=2, max_bytes=256, chunk_bytes=8)

    assert count == 1
