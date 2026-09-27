import json
from pathlib import Path

import pytest

from scripts.ops import release_freeze_guard as freeze


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _runner(*, dirty: bool = False, synchronized: bool = True):
    responses = {
        ("rev-parse", "--is-inside-work-tree"): (0, "true\n", ""),
        ("rev-parse", "HEAD"): (0, "a" * 40 + "\n", ""),
        ("branch", "--show-current"): (0, "release/test\n", ""),
        ("status", "--porcelain=v1", "--untracked-files=all"): (0, " M core/runtime.py\n" if dirty else "", ""),
        ("rev-list", "--left-right", "--count", "@{upstream}...HEAD"): (
            0,
            "0\t0\n" if synchronized else "0\t1\n",
            "",
        ),
        ("ls-files", "-s"): (0, "100644 deadbeef 0\tcore/runtime.py\n", ""),
        ("tag", "--points-at", "HEAD"): (0, "v1.2.3\n", ""),
    }

    def run(_root: Path, args: list[str]) -> tuple[int, str, str]:
        return responses.get(tuple(args), (1, "", "unexpected command"))

    return run


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts" / "release_ops.sh").write_text("#!/bin/zsh\n", encoding="utf-8")
    window = project / "governance" / "runtime" / "release_freeze_window.json"
    _write(
        window,
        {
            "active": True,
            "started_at_utc": "2026-08-01T00:00:00+00:00",
            "ends_at_utc": "2099-08-31T00:00:00+00:00",
            "reason": "paper_soak",
        },
    )
    return project, window


def test_clean_synchronized_commit_is_manifest_eligible(tmp_path: Path) -> None:
    project, window = _project(tmp_path)

    payload = freeze.build_payload(project, window_path=window, git_runner=_runner())

    assert payload["overall_status"] == "ready"
    assert payload["immutable_release_boundary"]["ready"] is True
    assert payload["git_integrity"]["tracked_tree_receipt_sha256"]
    assert payload["live_execution_authority"] is False


def test_dirty_tree_blocks_production_release_without_blocking_paper_soak(tmp_path: Path) -> None:
    project, window = _project(tmp_path)

    payload = freeze.build_payload(project, window_path=window, git_runner=_runner(dirty=True))

    assert payload["overall_status"] == "ready"
    assert payload["paper_soak_contract"]["ready"] is True
    assert payload["immutable_release_boundary"]["ready"] is False
    assert payload["git_integrity"]["changed_path_count"] == 1


def test_unsynchronized_commit_fails_release_integrity(tmp_path: Path) -> None:
    project, window = _project(tmp_path)

    payload = freeze.build_payload(project, window_path=window, git_runner=_runner(synchronized=False))

    assert payload["immutable_release_boundary"]["ready"] is False
    assert payload["git_integrity"]["ahead"] == 1


def test_manifest_binds_commit_tree_and_rollback() -> None:
    manifest = freeze._release_manifest(
        {
            "branch": "main",
            "commit": "b" * 40,
            "tracked_tree_receipt_sha256": "c" * 64,
            "tags_at_head": ["v2.0.0"],
        },
        window={"active": True, "reason": "release"},
    )

    assert manifest["release_identity"]["commit"] == "b" * 40
    assert manifest["rollback"]["reference"] == "b" * 40
    assert manifest["manifest_sha256"]


@pytest.mark.parametrize(
    "command",
    [
        ["status", "--porcelain=v1", "--untracked-files=all"],
        ["rev-parse", "HEAD"],
        ["ls-files", "-s"],
        ["rev-list", "--left-right", "--count", "@{upstream}...HEAD"],
    ],
)
def test_failed_git_probe_never_certifies_release(tmp_path, command):
    project, window = _project(tmp_path)
    good = _runner()

    def failed(root, args):
        return (1, "", "unavailable") if args == command else good(root, args)

    payload = freeze.build_payload(project, window_path=window, git_runner=failed)
    assert payload["git_integrity"]["ready"] is False
    assert payload["git_integrity"]["blockers"]
    assert payload["immutable_release_boundary"]["ready"] is False
    assert (
        payload["immutable_release_boundary"]["operator_release_review_required"]
        is True
    )


def test_malformed_upstream_counts_are_blocked_not_a_crash(tmp_path):
    project, window = _project(tmp_path)
    good = _runner()

    def malformed(root, args):
        return (0, "unknown unknown", "") if args[0] == "rev-list" else good(root, args)

    payload = freeze.build_payload(project, window_path=window, git_runner=malformed)
    assert (
        "git_upstream_state_unverified"
        in payload["immutable_release_boundary"]["blockers"]
    )


def test_expired_window_and_dirty_tree_have_separate_blockers(tmp_path):
    project, window = _project(tmp_path)
    _write(window, {"active": True, "ends_at_utc": "2020-01-01T00:00:00+00:00"})
    payload = freeze.build_payload(
        project, window_path=window, git_runner=_runner(dirty=True)
    )
    blockers = payload["immutable_release_boundary"]["blockers"]
    assert "release_freeze_window_inactive_or_expired" in blockers
    assert "release_worktree_not_clean" in blockers
    assert "release_upstream_not_synchronized" not in blockers
