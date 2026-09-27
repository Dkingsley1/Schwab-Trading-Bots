import argparse
import fcntl
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "governance" / "health" / "lock_watchdog_latest.json"

PID_RE = re.compile(r"pid=(\d+)")
POLICY_LOCK_NAMES = {"paper_trade.lock", "PAPER_TRADE_LOCK.flag"}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _extract_pid(text: str) -> int | None:
    m = PID_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _is_policy_lock(path: Path, text: str) -> bool:
    return path.name in POLICY_LOCK_NAMES or "live_data_paper_trade_only" in (
        text or ""
    )


def _lock_candidates() -> list[Path]:
    rows: list[Path] = []
    rows.extend(Path(PROJECT_ROOT / "governance").glob("*.lock"))
    rows.extend(Path(PROJECT_ROOT / "governance" / "locks").glob("*.lock"))
    uniq = {str(p): p for p in rows if not p.is_symlink() and p.is_file()}
    return [uniq[k] for k in sorted(uniq.keys())]


def _kernel_lock_state(path: Path) -> str:
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return "unknown"
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return "held"
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return "idle"
    except OSError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Observe lock ownership without unlinking kernel-lock anchors."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Compatibility flag; lock inodes are always preserved.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    stale: list[dict] = []
    healthy: list[dict] = []
    policy_locks: list[dict] = []
    idle_locks: list[dict] = []
    unknown_locks: list[dict] = []

    for path in _lock_candidates():
        text = ""
        try:
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
                text = handle.read(8192).decode("utf-8", errors="ignore")
        except Exception:
            unknown_locks.append(
                {"lock_path": str(path), "reason": "metadata_unavailable"}
            )
            continue

        if _is_policy_lock(path, text):
            policy_locks.append({"lock_path": str(path), "reason": "persistent_policy"})
            continue

        pid = _extract_pid(text)
        state = _kernel_lock_state(path)
        if state == "held":
            healthy.append(
                {"lock_path": str(path), "pid": pid, "reason": "kernel_lock_held"}
            )
            continue
        if state == "unknown":
            unknown_locks.append(
                {
                    "lock_path": str(path),
                    "reason": "kernel_lock_probe_unavailable",
                    "pid": pid,
                }
            )
            continue
        if pid is None:
            idle_locks.append(
                {
                    "lock_path": str(path),
                    "reason": "idle_kernel_lock_anchor",
                    "pid": None,
                }
            )
        elif _pid_alive(pid):
            healthy.append(
                {"lock_path": str(path), "pid": pid, "reason": "owner_pid_running"}
            )
        else:
            stale.append(
                {"lock_path": str(path), "reason": "owner_pid_not_running", "pid": pid}
            )

    # flock ownership ends when its handle closes. Unlinking even an idle anchor
    # can split ownership between an already-open inode and a replacement file.
    removed: list[str] = []

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "healthy_locks": healthy,
        "policy_locks": policy_locks,
        "idle_locks": idle_locks,
        "unknown_locks": unknown_locks,
        "stale_locks": stale,
        "lock_inode_preservation": True,
        "apply": bool(args.apply),
        "removed": removed,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"lock_watchdog stale={len(stale)} removed={len(removed)} healthy={len(healthy)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
