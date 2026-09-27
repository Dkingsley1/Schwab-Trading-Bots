import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ops/live_feed_tail.sh"


@pytest.mark.parametrize("heavy", ["0", "1"])
@pytest.mark.parametrize("visible", ["0", "1"])
def test_status_refresh_independent_of_connection_chatter(heavy, visible):
    source = SCRIPT.read_text()
    function = source.split("emit_live_feed_keepalive() {", 1)[1].split(
        "\nstart_heavy_ttl_guard()", 1
    )[0]
    command = """
SHOW_KEEPALIVE=$1 VISIBLE_KEEPALIVE_ALLOWED=$1 HEAVY_REQUESTED=$2
SOURCE=all IMPORTANT_ONLY=0 KEEPALIVE_SECONDS=15
STATUS_SNAPSHOT=1 KEEPALIVE_STATUS_EVERY=4
INCLUDE_DECISIONS=1 KEEPALIVE_DECISION_SNAPSHOT=1 KEEPALIVE_DECISION_EVERY=1
typeset -a files
emit_livefeed_status_snapshot() { print -r -- "$source_state"; }
emit_livefeed_decision_paper_snapshot() { print decision_scan; }
truncate_live_lines() { cat; }
"""
    command += "\nemit_live_feed_keepalive() {" + function
    command += "\nsource_state=old\nemit_live_feed_keepalive 4\nsource_state=new\nemit_live_feed_keepalive 8\n"
    result = subprocess.run(
        ["zsh", "-c", command, "test", visible, heavy],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "old\n" in result.stdout and "new\n" in result.stdout
    assert ("live_feed_keepalive timestamp_utc=" in result.stdout) == (visible == "1")
    assert ("decision_scan" in result.stdout) == (visible == "1" and heavy == "1")


def test_status_refresh_honors_explicit_disable():
    source = SCRIPT.read_text()
    function = source.split("emit_live_feed_keepalive() {", 1)[1].split(
        "\nstart_heavy_ttl_guard()", 1
    )[0]
    command = """
SHOW_KEEPALIVE=0 VISIBLE_KEEPALIVE_ALLOWED=0 HEAVY_REQUESTED=0
STATUS_SNAPSHOT=0 KEEPALIVE_STATUS_EVERY=4
emit_livefeed_status_snapshot() { print unexpected; }
truncate_live_lines() { cat; }
"""
    command += (
        "\nemit_live_feed_keepalive() {" + function + "\nemit_live_feed_keepalive 4\n"
    )
    result = subprocess.run(
        ["zsh", "-c", command], capture_output=True, text=True, check=True
    )
    assert "unexpected" not in result.stdout
