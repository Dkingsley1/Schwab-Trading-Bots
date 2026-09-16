import sqlite3
import subprocess
import sys
from pathlib import Path


def test_archive_inventory_and_vacuum_close_handles_without_garbage_collection(
    tmp_path,
):
    path = tmp_path / "archive ? quoted.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY)")
    conn.close()
    program = """
import gc, resource, sys
from pathlib import Path
from scripts.ops.cold_archive_compactor import _sqlite_inventory, _vacuum_sqlite
gc.disable()
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(64, soft), hard))
path = Path(sys.argv[1])
for _ in range(200):
    row = _sqlite_inventory(path)
    assert row['ok'], row
for _ in range(20):
    row = _vacuum_sqlite(path, {})
    assert row['status'] == 'vacuumed_verified', row
with (path.parent / 'completion.txt').open('w') as output:
    output.write('complete')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "completion.txt").read_text() == "complete"
