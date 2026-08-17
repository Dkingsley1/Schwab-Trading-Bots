#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import storage_router


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_split_brain_reconciler_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "storage" / "storage_split_brain_reconciler_latest.md"
TEXT_MERGE_SUFFIXES = {".jsonl", ".log", ".txt"}
ROUTER_REPAIR_PREFIXES = ("decision_explanations/", "decisions/", "logs/")
ROUTER_REPAIR_ARCHIVE_SUFFIXES = {".gz"}


class FailbackTimeout(TimeoutError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@contextlib.contextmanager
def _bounded_failback(seconds: int):
    timeout_sec = max(int(seconds), 1)
    if not hasattr(signal, "setitimer"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _handler(_signum, _frame) -> None:
        raise FailbackTimeout(f"storage route failback timed out after {timeout_sec}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _force_failback(project_root: Path, *, timeout_sec: int, fallback_conflicts: int) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    os.environ["BOT_LOGS_PREFER_EXTERNAL"] = "1"
    try:
        with _bounded_failback(timeout_sec):
            routing = storage_router.route_runtime_storage(project_root)
    except FailbackTimeout as exc:
        duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
        return {
            "attempted": True,
            "ok": False,
            "timed_out": True,
            "timeout_sec": max(int(timeout_sec), 1),
            "duration_ms": duration_ms,
            "mode": "",
            "active_root": "",
            "split_brain_conflicts": int(fallback_conflicts),
            "error": str(exc),
        }
    except Exception as exc:
        duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
        return {
            "attempted": True,
            "ok": False,
            "timed_out": False,
            "timeout_sec": max(int(timeout_sec), 1),
            "duration_ms": duration_ms,
            "mode": "",
            "active_root": "",
            "split_brain_conflicts": int(fallback_conflicts),
            "error": f"{type(exc).__name__}: {exc}",
        }

    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    split_brain_conflicts = int(getattr(routing, "split_brain_conflicts", fallback_conflicts) or 0)
    return {
        "attempted": True,
        "ok": split_brain_conflicts == 0,
        "timed_out": False,
        "timeout_sec": max(int(timeout_sec), 1),
        "duration_ms": duration_ms,
        "mode": str(getattr(routing, "mode", "")),
        "active_root": str(getattr(routing, "active_root", "")),
        "split_brain_conflicts": split_brain_conflicts,
        "error": "",
    }


def _local_fallback_root(project_root: Path) -> Path:
    return Path(os.getenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(project_root / storage_router.DEFAULT_LOCAL_FALLBACK))).expanduser()


def _strip_conflict_suffix(path: Path) -> Path:
    name = path.name
    if ".local_fallback" not in name:
        return path
    base = name.split(".local_fallback", 1)[0]
    return path.with_name(base)


def _iter_conflict_files(external_root: Path) -> list[Path]:
    if not external_root.exists():
        return []
    matches: list[Path] = []
    for root, _, files in os.walk(external_root):
        for fname in files:
            if ".local_fallback" in fname:
                matches.append(Path(root) / fname)
    return sorted(matches)


def _safe_stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace(":", "").replace("+", "_")


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    seq = 1
    while True:
        candidate = path.with_name(f"{path.name}.dupe.{seq}")
        if not candidate.exists():
            return candidate
        seq += 1


def _archive_root(project_root: Path, *, stamp: str | None = None) -> Path:
    return project_root / "local_fallback_storage" / "quarantine" / "storage_split_brain" / (stamp or _safe_stamp())


def _write_manifest(archive_root: Path, actions: list[dict[str, Any]]) -> str:
    manifest_path = archive_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action_count": len(actions),
        "actions": actions,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return str(manifest_path)


def _archive_file(path: Path, *, source_root: Path, archive_base: Path, action: str, actions: list[dict[str, Any]]) -> bool:
    if not path.exists() or not path.is_file():
        actions.append({"action": action, "source": str(path), "status": "missing"})
        return False
    try:
        rel_path = path.relative_to(source_root)
    except Exception:
        rel_path = Path(path.name)
    destination = _unique_destination(archive_base / rel_path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
    except Exception as exc:
        actions.append({"action": action, "source": str(path), "target": str(destination), "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return False
    actions.append({"action": action, "source": str(path), "target": str(destination), "status": "archived"})
    return True


def _archive_external_conflict_sidecars(
    rows: list[dict[str, Any]],
    *,
    external_root: Path,
    archive_base: Path,
    apply: bool,
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    del rows
    selected = [{"conflict_file": str(path)} for path in _iter_conflict_files(external_root)]
    if not apply:
        return {
            "requested": True,
            "applied": False,
            "candidate_count": len(selected),
            "archived_count": 0,
            "actions": [],
        }
    for row in selected:
        conflict_path = Path(str(row.get("conflict_file") or "")).expanduser()
        _archive_file(
            conflict_path,
            source_root=external_root,
            archive_base=archive_base / "external_sidecars",
            action="archive_external_local_fallback_sidecar",
            actions=actions,
        )
    return {
        "requested": True,
        "applied": True,
        "candidate_count": len(selected),
        "archived_count": sum(1 for row in actions if row.get("status") == "archived"),
        "actions": actions,
    }


def _router_log_conflict_allowed(rel_path: str, local_path: Path) -> bool:
    rel_norm = str(rel_path or "").replace("\\", "/").lstrip("./")
    if not rel_norm.startswith(ROUTER_REPAIR_PREFIXES):
        return False
    if ".local_fallback" in Path(rel_norm).name:
        return False
    if not local_path.exists() or not local_path.is_file() or local_path.is_symlink():
        return False
    suffix = local_path.suffix.lower()
    return suffix in TEXT_MERGE_SUFFIXES or suffix in ROUTER_REPAIR_ARCHIVE_SUFFIXES


def _router_log_conflict_rows(project_root: Path) -> list[dict[str, Any]]:
    local_root = _local_fallback_root(project_root)
    external_root = storage_router._external_project_root()
    local_sig = storage_router._scan_tree_signature(local_root, storage_router.DEFAULT_LINK_DIRS, max_files=5000)
    if not local_sig:
        return []
    external_sig = storage_router._scan_tree_signature(external_root, storage_router.DEFAULT_LINK_DIRS, max_files=5000)
    rows: list[dict[str, Any]] = []
    for rel_path in sorted(local_sig):
        external_meta = external_sig.get(rel_path)
        if not external_meta or external_meta == local_sig[rel_path]:
            continue
        local_path = local_root / rel_path
        external_path = external_root / rel_path
        rows.append(
            {
                "relative_path": rel_path,
                "local_path": str(local_path),
                "external_path": str(external_path),
                "local_meta": local_sig[rel_path],
                "external_meta": external_meta,
                "repair_allowed": _router_log_conflict_allowed(rel_path, local_path),
            }
        )
    return rows


def _append_then_archive_local_log(
    local_path: Path,
    external_path: Path,
    *,
    local_root: Path,
    archive_base: Path,
    actions: list[dict[str, Any]],
) -> bool:
    if not local_path.exists() or not local_path.is_file():
        actions.append({"action": "merge_local_log_to_external", "source": str(local_path), "target": str(external_path), "status": "missing"})
        return False
    archived = _archive_file(
        local_path,
        source_root=local_root,
        archive_base=archive_base / "local_router_conflicts",
        action="archive_local_router_conflict_before_merge",
        actions=actions,
    )
    if not archived:
        return False

    archive_target = Path(str(actions[-1].get("target") or ""))
    try:
        external_path.parent.mkdir(parents=True, exist_ok=True)
        appended_lines = 0
        skipped_duplicate_lines = 0
        if archive_target.suffix.lower() == ".jsonl":
            existing_lines: set[bytes] = set()
            if external_path.exists():
                with external_path.open("rb") as existing_handle:
                    for raw in existing_handle:
                        line = raw.rstrip(b"\r\n")
                        if line:
                            existing_lines.add(line)
            with external_path.open("ab") as out_handle, archive_target.open("rb") as in_handle:
                for raw in in_handle:
                    line = raw.rstrip(b"\r\n")
                    if not line:
                        continue
                    if line in existing_lines:
                        skipped_duplicate_lines += 1
                        continue
                    out_handle.write(line + b"\n")
                    existing_lines.add(line)
                    appended_lines += 1
        else:
            with external_path.open("ab") as out_handle, archive_target.open("rb") as in_handle:
                if external_path.exists() and external_path.stat().st_size > 0:
                    out_handle.write(b"\n")
                shutil.copyfileobj(in_handle, out_handle)
    except Exception as exc:
        actions.append({"action": "merge_local_log_to_external", "source": str(archive_target), "target": str(external_path), "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return False
    actions.append(
        {
            "action": "merge_local_log_to_external",
            "source": str(archive_target),
            "target": str(external_path),
            "status": "merged",
            "appended_lines": appended_lines,
            "skipped_duplicate_lines": skipped_duplicate_lines,
        }
    )
    return True


def _repair_router_log_conflicts(project_root: Path, *, archive_base: Path, apply: bool) -> dict[str, Any]:
    rows = _router_log_conflict_rows(project_root)
    local_root = _local_fallback_root(project_root)
    external_root = storage_router._external_project_root()
    actions: list[dict[str, Any]] = []
    allowed = [row for row in rows if bool(row.get("repair_allowed", False))]
    blocked = [row for row in rows if not bool(row.get("repair_allowed", False))]
    if not apply:
        return {
            "requested": True,
            "applied": False,
            "candidate_count": len(rows),
            "allowed_count": len(allowed),
            "blocked_count": len(blocked),
            "repaired_count": 0,
            "blocked": blocked,
            "actions": [],
        }

    repaired = 0
    for row in allowed:
        local_path = Path(str(row.get("local_path") or "")).expanduser()
        external_path = Path(str(row.get("external_path") or "")).expanduser()
        suffix = local_path.suffix.lower()
        ok = False
        if suffix in TEXT_MERGE_SUFFIXES:
            ok = _append_then_archive_local_log(
                local_path,
                external_path,
                local_root=local_root,
                archive_base=archive_base,
                actions=actions,
            )
        elif suffix in ROUTER_REPAIR_ARCHIVE_SUFFIXES:
            ok = _archive_file(
                local_path,
                source_root=local_root,
                archive_base=archive_base / "local_router_conflicts",
                action="archive_local_router_conflict",
                actions=actions,
            )
        if ok:
            repaired += 1

    return {
        "requested": True,
        "applied": True,
        "candidate_count": len(rows),
        "allowed_count": len(allowed),
        "blocked_count": len(blocked),
        "repaired_count": repaired,
        "blocked": blocked,
        "actions": actions,
        "local_root": str(local_root),
        "external_root": str(external_root),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Storage Split-Brain Reconciliation",
        "",
        f"- Timestamp UTC: `{payload.get('timestamp_utc', '')}`",
        f"- External Root: `{payload.get('external_root', '')}`",
        f"- Local Fallback Root: `{payload.get('local_root', '')}`",
        f"- Conflict Files: `{int(summary.get('conflict_files', 0) or 0)}`",
        f"- Hash Match Ready: `{int(summary.get('hash_match_ready', 0) or 0)}`",
        f"- Unresolved Conflicts: `{int(summary.get('unresolved_conflicts', 0) or 0)}`",
        f"- Router Conflicts: `{int(summary.get('router_conflicts', 0) or 0)}`",
        f"- Router Repairable Conflicts: `{int(summary.get('router_repairable_conflicts', 0) or 0)}`",
        f"- Force Failback Eligible: `{bool(summary.get('force_failback_eligible', False))}`",
        "",
        "## Top Conflicts",
        "",
    ]
    for row in (payload.get("conflicts") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- `{rel}`: class=`{cls}` local_hash_match=`{match}`".format(
                rel=str(row.get("relative_path") or ""),
                cls=str(row.get("classification") or ""),
                match=bool(row.get("hashes_match", False)),
            )
        )
    lines.extend(["", "## Router Conflicts", ""])
    for row in (payload.get("router_conflicts") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- `{rel}`: repair_allowed=`{allowed}`".format(
                rel=str(row.get("relative_path") or ""),
                allowed=bool(row.get("repair_allowed", False)),
            )
        )
    return "\n".join(lines) + "\n"


def build_payload(project_root: Path = PROJECT_ROOT, *, full_scan: bool = False) -> dict[str, Any]:
    external_root = storage_router._external_project_root()
    local_root = _local_fallback_root(project_root)
    failback_payload = _load_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json")
    mount_payload = _load_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json")
    router_conflicts = _router_log_conflict_rows(project_root)

    artifact_reported_split_brain_conflicts = max(
        int(failback_payload.get("split_brain_conflicts", 0) or 0),
        int(mount_payload.get("storage_mount_transition", {}).get("recovery", {}).get("payload", {}).get("split_brain_conflicts", 0) or 0),
    )
    scan_mode = "full_scan" if full_scan else "manifest_fast_path"
    rows: list[dict[str, Any]] = []
    if full_scan or artifact_reported_split_brain_conflicts > 0:
        for conflict_path in _iter_conflict_files(external_root):
            canonical_external = _strip_conflict_suffix(conflict_path)
            relative_path = str(canonical_external.relative_to(external_root)) if canonical_external.exists() or external_root.exists() else canonical_external.name
            local_counterpart = local_root / relative_path
            hashes_match = False
            local_hash = ""
            external_hash = ""
            conflict_hash = ""
            if conflict_path.exists():
                conflict_hash = _sha(conflict_path)
            if local_counterpart.exists() and local_counterpart.is_file():
                local_hash = _sha(local_counterpart)
            if canonical_external.exists() and canonical_external.is_file():
                external_hash = _sha(canonical_external)
            if local_hash and external_hash and local_hash == external_hash:
                hashes_match = True
                classification = "ready_to_prune_local"
            elif conflict_hash and external_hash and conflict_hash == external_hash:
                hashes_match = True
                classification = "duplicate_conflict_copy"
            elif not canonical_external.exists():
                classification = "external_missing_keep_local"
            elif not local_counterpart.exists():
                classification = "conflict_copy_only_review"
            else:
                classification = "divergent_hash_manual_review"
            rows.append(
                {
                    "conflict_file": str(conflict_path),
                    "relative_path": relative_path,
                    "external_path": str(canonical_external),
                    "local_path": str(local_counterpart),
                    "classification": classification,
                    "hashes_match": hashes_match,
                    "local_hash": local_hash[:16],
                    "external_hash": external_hash[:16],
                    "conflict_hash": conflict_hash[:16],
                }
            )

    rows.sort(key=lambda row: (str(row.get("classification") or ""), str(row.get("relative_path") or "")))
    hash_match_ready = sum(1 for row in rows if bool(row.get("hashes_match", False)))
    unresolved_conflicts = sum(1 for row in rows if not bool(row.get("hashes_match", False)))
    concrete_blocking_conflicts = max(unresolved_conflicts, len(router_conflicts))
    stale_reported_conflicts_suppressed = 0
    if artifact_reported_split_brain_conflicts > 0 and concrete_blocking_conflicts == 0:
        split_brain_conflicts = 0
        stale_reported_conflicts_suppressed = artifact_reported_split_brain_conflicts
    else:
        split_brain_conflicts = max(
            artifact_reported_split_brain_conflicts,
            unresolved_conflicts,
            len(router_conflicts),
        )
    force_failback_eligible = bool(mount_payload.get("external_available", False)) and split_brain_conflicts == 0

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "external_root": str(external_root),
        "local_root": str(local_root),
        "external_available": bool(mount_payload.get("external_available", False)),
        "current_storage_mode": str(mount_payload.get("storage_mode") or failback_payload.get("mode") or ""),
        "scan_mode": scan_mode,
        "summary": {
            "conflict_files": len(rows),
            "hash_match_ready": hash_match_ready,
            "unresolved_conflicts": unresolved_conflicts,
            "router_conflicts": len(router_conflicts),
            "router_repairable_conflicts": sum(1 for row in router_conflicts if bool(row.get("repair_allowed", False))),
            "artifact_reported_split_brain_conflicts": artifact_reported_split_brain_conflicts,
            "stale_reported_split_brain_conflicts_suppressed": stale_reported_conflicts_suppressed,
            "reported_split_brain_conflicts": split_brain_conflicts,
            "force_failback_eligible": force_failback_eligible,
        },
        "conflicts": rows[:50],
        "router_conflicts": router_conflicts[:50],
        "recommended_actions": [
            "prune duplicate local fallback copies when hashes already match external storage",
            "archive/merge log-class router conflicts before asking storage to fail back to BOT_LOGS",
            "hold force failback until unresolved_conflicts reaches zero",
            "treat bounded failback timeout as degraded storage recovery, not a wedged soak",
            "use the markdown report to review divergent hashes before deleting any conflict copies",
            "run with --full-scan when you want to sweep the entire external tree for forensic reconciliation",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify and report BOT_LOGS split-brain conflicts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--archive-external-conflict-sidecars", action="store_true")
    parser.add_argument("--repair-router-log-conflicts", action="store_true")
    parser.add_argument("--force-failback-if-hashes-match", action="store_true")
    parser.add_argument("--force-failback-timeout-sec", type=int, default=45)
    parser.add_argument("--full-scan", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root, full_scan=bool(args.full_scan))
    repair_requested = bool(args.archive_external_conflict_sidecars or args.repair_router_log_conflicts)
    repair_result: dict[str, Any] = {
        "requested": repair_requested,
        "applied": bool(args.apply and repair_requested),
    }
    if repair_requested:
        stamp = _safe_stamp()
        archive_base = _archive_root(project_root, stamp=stamp)
        before_summary = dict(payload.get("summary") or {})
        repair_actions: list[dict[str, Any]] = []
        if args.archive_external_conflict_sidecars:
            sidecar_result = _archive_external_conflict_sidecars(
                list(payload.get("conflicts") or []),
                external_root=storage_router._external_project_root(),
                archive_base=archive_base,
                apply=bool(args.apply),
            )
            repair_result["external_sidecars"] = sidecar_result
            repair_actions.extend(list(sidecar_result.get("actions") or []))
        if args.repair_router_log_conflicts:
            router_result = _repair_router_log_conflicts(project_root, archive_base=archive_base, apply=bool(args.apply))
            repair_result["router_log_conflicts"] = router_result
            repair_actions.extend(list(router_result.get("actions") or []))
        if args.apply:
            repair_result["archive_root"] = str(archive_base)
            repair_result["manifest_path"] = _write_manifest(archive_base, repair_actions)
            payload = build_payload(project_root, full_scan=True)
            repair_result["before_summary"] = before_summary
            repair_result["after_summary"] = dict(payload.get("summary") or {})
        payload["repair"] = repair_result

    if args.force_failback_if_hashes_match and bool(payload.get("summary", {}).get("force_failback_eligible", False)):
        payload["forced_failback"] = _force_failback(
            project_root,
            timeout_sec=int(args.force_failback_timeout_sec),
            fallback_conflicts=int(payload.get("summary", {}).get("reported_split_brain_conflicts", 0) or 0),
        )
    else:
        payload["forced_failback"] = {
            "attempted": bool(args.force_failback_if_hashes_match),
            "ok": False,
            "timed_out": False,
            "timeout_sec": max(int(args.force_failback_timeout_sec), 1),
            "mode": "",
            "active_root": "",
            "split_brain_conflicts": int(payload.get("summary", {}).get("reported_split_brain_conflicts", 0) or 0),
            "error": "",
        }

    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    markdown_path = Path(args.markdown_out).expanduser()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_split_brain_reconciler "
            f"unresolved_conflicts={int(payload.get('summary', {}).get('unresolved_conflicts', 0) or 0)} "
            f"force_failback_eligible={bool(payload.get('summary', {}).get('force_failback_eligible', False))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
