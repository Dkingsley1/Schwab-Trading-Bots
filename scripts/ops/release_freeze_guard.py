#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, utc_now, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, utc_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "release_freeze_guard_latest.json"
DEFAULT_WINDOW_PATH = PROJECT_ROOT / "governance" / "runtime" / "release_freeze_window.json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "governance" / "releases" / "immutable_release_manifest_latest.json"
GitRunner = Callable[[Path, list[str]], tuple[int, str, str]]


def _load_window(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload:
        return payload
    return {"active": False}


def _save_window(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _default_git_runner(project_root: Path, args: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 124, "", str(exc)
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")


def _git_snapshot(project_root: Path, *, runner: GitRunner = _default_git_runner) -> dict[str, Any]:
    def run(*args: str) -> tuple[int, str, str]:
        return runner(project_root, list(args))

    worktree_rc, worktree_out, worktree_err = run("rev-parse", "--is-inside-work-tree")
    repository = bool(worktree_rc == 0 and worktree_out.strip() == "true")
    if not repository:
        return {
            "repository": False,
            "ready": False,
            "error": worktree_err.strip() or "not_a_git_worktree",
            "branch": "",
            "commit": "",
            "clean": False,
            "changed_path_count": 0,
            "changed_paths": [],
            "upstream_configured": False,
            "ahead": None,
            "behind": None,
            "upstream_synchronized": False,
            "tracked_tree_receipt_sha256": "",
            "tags_at_head": [],
        }

    _, commit_out, _ = run("rev-parse", "HEAD")
    _, branch_out, _ = run("branch", "--show-current")
    _, status_out, _ = run("status", "--porcelain=v1", "--untracked-files=all")
    changed_paths = [line.rstrip() for line in status_out.splitlines() if line.strip()]
    upstream_rc, upstream_out, _ = run("rev-list", "--left-right", "--count", "@{upstream}...HEAD")
    ahead: int | None = None
    behind: int | None = None
    if upstream_rc == 0:
        values = upstream_out.strip().split()
        if len(values) == 2:
            behind, ahead = int(values[0]), int(values[1])
    _, tree_out, _ = run("ls-files", "-s")
    _, tags_out, _ = run("tag", "--points-at", "HEAD")
    tree_receipt = hashlib.sha256(tree_out.encode("utf-8")).hexdigest() if tree_out else ""
    clean = not changed_paths
    upstream_synchronized = bool(upstream_rc == 0 and ahead == 0 and behind == 0)
    ready = bool(commit_out.strip() and clean and upstream_synchronized and tree_receipt)
    return {
        "repository": True,
        "ready": ready,
        "error": "",
        "branch": branch_out.strip(),
        "commit": commit_out.strip(),
        "clean": clean,
        "changed_path_count": len(changed_paths),
        "changed_paths": changed_paths[:40],
        "changed_paths_truncated": len(changed_paths) > 40,
        "upstream_configured": upstream_rc == 0,
        "ahead": ahead,
        "behind": behind,
        "upstream_synchronized": upstream_synchronized,
        "tracked_tree_receipt_sha256": tree_receipt,
        "tags_at_head": sorted(line.strip() for line in tags_out.splitlines() if line.strip()),
    }


def _release_manifest(git_integrity: dict[str, Any], *, window: dict[str, Any]) -> dict[str, Any]:
    base = {
        "schema_version": 1,
        "created_at_utc": iso_now(),
        "release_identity": {
            "branch": str(git_integrity.get("branch") or ""),
            "commit": str(git_integrity.get("commit") or ""),
            "tracked_tree_receipt_sha256": str(git_integrity.get("tracked_tree_receipt_sha256") or ""),
            "tags_at_head": list(git_integrity.get("tags_at_head") or []),
        },
        "rollback": {
            "reference": str(git_integrity.get("commit") or ""),
            "command": f"./scripts/release_ops.sh rollback {git_integrity.get('commit')}" if git_integrity.get("commit") else "",
        },
        "freeze_window": {
            "active": bool(window.get("active", False)),
            "started_at_utc": str(window.get("started_at_utc") or ""),
            "ends_at_utc": str(window.get("ends_at_utc") or ""),
            "reason": str(window.get("reason") or ""),
        },
        "live_execution_authority": False,
    }
    receipt = hashlib.sha256(
        json.dumps(base, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**base, "manifest_sha256": receipt}


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    window_path: Path = DEFAULT_WINDOW_PATH,
    git_runner: GitRunner = _default_git_runner,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    promotion_readiness = load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json")
    new_bot_graduation = load_json(project_root / "governance" / "walk_forward" / "new_bot_graduation_latest.json")
    supportability_control = load_json(health_root / "supportability_control_latest.json")
    window = _load_window(window_path)
    ends_at = parse_iso_utc(window.get("ends_at_utc"))
    active = bool(window.get("active", False)) and (ends_at is None or ends_at > utc_now())
    git_integrity = _git_snapshot(project_root, runner=git_runner)
    rollback_entrypoint = project_root / "scripts" / "release_ops.sh"
    rollback_ready = bool(git_integrity.get("commit") and rollback_entrypoint.is_file())
    production_release_ready = bool(active and git_integrity.get("ready", False) and rollback_ready)

    overall_status = "ready" if active else "degraded"
    if active and str(supportability_control.get("overall_status") or "") == "blocked":
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "./scripts/ops/opsctl.sh release-freeze --activate-days 21 --reason multi_week_runtime_window --json" if not active else "",
            "hold promotions, schema churn, and experimental bot activations while the long-run window is active" if active else "",
            "only thaw the window after promotion readiness, supportability, and freshness lanes are back inside budget" if active else "",
            "commit or intentionally discard every worktree change before creating a production release" if not git_integrity.get("clean", False) else "",
            "synchronize the release commit with its configured upstream before production promotion" if not git_integrity.get("upstream_synchronized", False) else "",
            "restore scripts/release_ops.sh before production promotion" if not rollback_entrypoint.is_file() else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "window": {
            "active": active,
            "started_at_utc": str(window.get("started_at_utc") or ""),
            "ends_at_utc": str(window.get("ends_at_utc") or ""),
            "reason": str(window.get("reason") or ""),
        },
        "frozen_surfaces": {
            "allow_promotions": not active,
            "allow_schema_changes": not active,
            "allow_experimental_bots": not active,
        },
        "git_integrity": git_integrity,
        "immutable_release_boundary": {
            "ready": production_release_ready,
            "status": "ready" if production_release_ready else "blocked",
            "rollback_ready": rollback_ready,
            "rollback_reference": str(git_integrity.get("commit") or ""),
            "rollback_entrypoint": str(rollback_entrypoint) if rollback_entrypoint.is_file() else "",
            "manifest_eligible": production_release_ready,
            "requires_clean_worktree": True,
            "requires_upstream_synchronization": True,
        },
        "paper_soak_contract": {
            "ready": overall_status == "ready",
            "release_integrity_debt_blocks_paper_collection": False,
            "release_integrity_debt_blocks_live_promotion": True,
        },
        "gating_context": {
            "promotion_ready": bool(promotion_readiness.get("promote_ok", False)),
            "new_bot_graduation_ok": bool(new_bot_graduation.get("ok", False)),
            "supportability_status": str(supportability_control.get("overall_status") or ""),
        },
        "infra_bots": ["release_freeze_guard", "promotion_readiness_summary", "supportability_control"],
        "live_execution_authority": False,
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage release freeze windows for long runtime runs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--window-path")
    parser.add_argument("--activate-days", type=int, default=0)
    parser.add_argument("--clear-window", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument("--out-file")
    parser.add_argument("--write-release-manifest", action="store_true")
    parser.add_argument("--manifest-file")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    window_path = (
        Path(args.window_path).expanduser()
        if args.window_path
        else project_root / "governance" / "soak" / "release_freeze_window.json"
    )
    if not window_path.is_absolute():
        window_path = project_root / window_path
    if args.clear_window:
        _save_window(window_path, {"active": False, "cleared_at_utc": iso_now(), "reason": str(args.reason or "")})
    elif int(args.activate_days) > 0:
        started = utc_now()
        payload = {
            "active": True,
            "started_at_utc": started.isoformat(),
            "ends_at_utc": (started + timedelta(days=int(args.activate_days))).isoformat(),
            "reason": str(args.reason or "runtime_freeze_window"),
        }
        _save_window(window_path, payload)

    payload = build_payload(project_root, window_path=window_path)
    manifest_path = (
        Path(args.manifest_file).expanduser()
        if args.manifest_file
        else project_root / "governance" / "releases" / "immutable_release_manifest_latest.json"
    )
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_written = False
    if args.write_release_manifest and bool((payload.get("immutable_release_boundary") or {}).get("ready", False)):
        manifest = _release_manifest(payload["git_integrity"], window=payload["window"])
        write_payload(manifest_path, manifest)
        manifest_written = True
    payload["immutable_release_boundary"]["manifest_path"] = str(manifest_path)
    payload["immutable_release_boundary"]["manifest_written"] = manifest_written
    out_path = (
        Path(args.out_file).expanduser()
        if args.out_file
        else project_root / "governance" / "health" / "release_freeze_guard_latest.json"
    )
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "release_freeze_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"active={int(bool(((payload.get('window') or {}).get('active', False))))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
