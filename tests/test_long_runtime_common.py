import json
from pathlib import Path

from scripts.ops.long_runtime_common import write_payload


def test_write_payload_atomically_replaces_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "health" / "artifact.json"
    write_payload(path, {"generation": 1, "grade": "A++"})
    first_inode = path.stat().st_ino

    write_payload(path, {"generation": 2, "grade": "A++"})

    assert path.stat().st_ino != first_inode
    assert json.loads(path.read_text(encoding="utf-8")) == {"generation": 2, "grade": "A+"}
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_write_payload_preserves_existing_mode(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_payload(path, {"generation": 1})
    path.chmod(0o640)

    write_payload(path, {"generation": 2})

    assert path.stat().st_mode & 0o777 == 0o640
