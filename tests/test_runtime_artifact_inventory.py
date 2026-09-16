import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/releases/2026-09-16-runtime-artifact-inventory.json"


@pytest.fixture(scope="module")
def inventory():
    return json.loads(INVENTORY.read_text())


def test_inventory_accounts_for_every_reviewed_entry(inventory):
    entries = inventory["entries"]
    assert inventory["total_entries"] == len(entries) == 361
    assert len({row["path"] for row in entries}) == 361
    assert inventory["counts"] == dict(Counter(row["kind"] for row in entries))
    assert inventory["counts"] == {
        "registry_backup": 21,
        "backup_link": 51,
        "runtime_report": 289,
    }
    assert inventory["dispositions"] == dict(
        Counter(row["disposition"] for row in entries)
    )
    assert inventory["dispositions"] == {
        "retain_registry_backup": 21,
        "retain_backup_lookup_link": 51,
        "retain_local_historical_report": 288,
        "removed_verified_empty_report": 1,
    }
    assert inventory["report_bytes_at_review"] == sum(
        row["size_bytes"] for row in entries if row["kind"] == "runtime_report"
    )


def test_inventory_is_payload_free_and_has_honest_hash_scopes(inventory):
    assert inventory["schema_version"] == 1
    assert re.fullmatch(r"[a-f0-9]{40}", inventory["reviewed_source_commit"])
    assert inventory["privacy"] == {
        "raw_payloads_included": False,
        "link_destinations_included": False,
    }
    assert inventory["authority"] == {
        "archive_payloads_verified": False,
        "external_targets_inspected": False,
        "restore_readiness": False,
        "live_execution_authority": False,
    }
    scopes = {
        "backup_link": "link_text_only",
        "registry_backup": "compressed_file_bytes",
        "runtime_report": "file_bytes",
    }
    for row in inventory["entries"]:
        assert set(row) == {
            "path",
            "kind",
            "size_bytes",
            "sha256",
            "hash_scope",
            "disposition",
        }
        path = PurePosixPath(row["path"])
        assert not path.is_absolute() and ".." not in path.parts
        assert len(path.parts) == 2 and path.parts[0] in {"work", "backups"}
        assert re.fullmatch(r"[a-f0-9]{64}", row["sha256"])
        assert row["size_bytes"] >= 0
        assert row["hash_scope"] == scopes[row["kind"]]
        if row["disposition"] == "removed_verified_empty_report":
            assert row["path"] == "work/backpressure_super_drainer_targeted.json"
            assert row["size_bytes"] == 0
            assert row["sha256"] == hashlib.sha256(b"").hexdigest()
    assert "/Users/" not in INVENTORY.read_text()
    assert "/Volumes/" not in INVENTORY.read_text()


@pytest.fixture()
def isolated_git(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    return tmp_path


def test_all_reviewed_runtime_paths_are_excluded_without_target_reads(
    inventory, isolated_git
):
    paths = [row["path"] for row in inventory["entries"]]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=isolated_git,
        input=("\0".join(paths) + "\0").encode(),
        capture_output=True,
        check=True,
    )
    assert result.stdout.decode().split("\0")[:-1] == paths


@pytest.mark.parametrize(
    "path",
    [
        "work/new_report.json",
        "work/new_report.log",
        "work/last_issues_writer_sample.txt",
        "backups/master_bot_registry_before_future_review_20990101_010101.json.gz",
    ],
)
def test_named_runtime_output_families_stay_local(isolated_git, path):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", path],
        cwd=isolated_git,
        capture_output=True,
    )
    assert result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "work/new_source.py",
        "work/operator_notes.md",
        "work/new_sample.txt",
        "work/nested/source.json",
        "config/new_policy.json",
        "scripts/ops/new_owner.py",
        "tests/test_new_owner.py",
        "backups/README.md",
        "backups/unrelated.json.gz",
        "docs/releases/2026-09-16-runtime-artifact-inventory.json",
    ],
)
def test_source_and_unreviewed_families_remain_visible(isolated_git, path):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", path],
        cwd=isolated_git,
        capture_output=True,
    )
    assert result.returncode == 1
