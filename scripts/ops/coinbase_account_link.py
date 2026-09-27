#!/usr/bin/env python3
"""Documented Coinbase auth flow; credentials and holdings never go to stdout."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.coinbase_account import (
    CoinbaseAccountCredentials,
    CoinbaseAccountError,
    CoinbaseReadOnlyClient,
)

DEFAULT_STATE_DIR = (
    Path.home() / "Library/Application Support/SchwabTradingPlatform/coinbase-readonly"
)
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_STATUS_AGE_SECONDS = 300


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_path(path: Path) -> None:
    # Reject protected paths lexically, before any filesystem operation.
    absolute = Path(os.path.abspath(path.expanduser()))
    if absolute == Path("/Volumes/VIDEO") or Path("/Volumes/VIDEO") in absolute.parents:
        raise CoinbaseAccountError("protected_path")
    for parent in reversed((absolute, *absolute.parents)):
        if parent.is_symlink():
            raise CoinbaseAccountError("symlink_path_rejected")


def _read_json(path: Path, *, downloaded: bool = False) -> dict:
    _check_path(path)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        with os.fdopen(os.open(path, flags), "rb") as handle:
            info = os.fstat(handle.fileno())
            forbidden = 0o022 if downloaded else 0o077
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & forbidden
                or info.st_nlink != 1
            ):
                raise CoinbaseAccountError("unsafe_file_permissions")
            if downloaded:
                os.fchmod(handle.fileno(), 0o600)
            data = handle.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise CoinbaseAccountError("file_too_large")
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError()
        return payload
    except (ValueError, UnicodeError):
        raise CoinbaseAccountError("invalid_local_json") from None
    except FileNotFoundError:
        raise CoinbaseAccountError("credential_file_missing") from None
    except OSError:
        raise CoinbaseAccountError("local_file_unavailable") from None


def _private_dir(path: Path) -> None:
    _check_path(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.stat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise CoinbaseAccountError("unsafe_state_directory")


def _write_json(path: Path, payload: dict) -> None:
    _check_path(path)
    fd, temporary = tempfile.mkstemp(prefix=".coinbase-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def status(state_dir: Path = DEFAULT_STATE_DIR) -> dict:
    result = {
        "timestamp_utc": _now(),
        "overall_status": "not_linked",
        "connected": False,
        "live_execution_allowed": False,
        "transfers_allowed": False,
        "mode": "read_only",
        "account_count": None,
    }
    try:
        receipt = _read_json(state_dir / "status.json")
        measured = datetime.fromisoformat(receipt["verified_at"])
        age = (datetime.now(timezone.utc) - measured).total_seconds()
        result["verified_at"] = receipt["verified_at"]
        result["evidence_age_seconds"] = round(age, 1)
        if receipt.get("overall_status") != "ready":
            result["overall_status"] = "blocked"
        elif not 0 <= age <= MAX_STATUS_AGE_SECONDS:
            result["overall_status"] = "stale"
        else:
            connection = _read_json(state_dir / "connection.json")
            if connection.get("verified_at") != receipt["verified_at"]:
                raise ValueError()
            result.update(
                overall_status="ready",
                connected=True,
                account_count=receipt.get("account_count"),
            )
    except (CoinbaseAccountError, KeyError, TypeError, ValueError):
        pass
    return result


def sync_account(
    *,
    state_dir: Path = DEFAULT_STATE_DIR,
    key_file: Path | None = None,
    replace: bool = False,
    client_factory=CoinbaseReadOnlyClient,
) -> dict:
    result = {
        "timestamp_utc": _now(),
        "overall_status": "blocked",
        "connected": False,
        "live_execution_allowed": False,
        "transfers_allowed": False,
        "mode": "read_only",
        "account_count": None,
    }
    lock_fd = None
    locked = False
    try:
        _private_dir(state_dir)
        _check_path(state_dir / ".lock")
        lock_fd = os.open(
            state_dir / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        lock_info = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_uid != os.getuid()
            or lock_info.st_mode & 0o077
            or lock_info.st_nlink != 1
        ):
            raise CoinbaseAccountError("unsafe_lock_file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        connection_path = state_dir / "connection.json"
        _check_path(connection_path)
        if key_file is not None:
            if connection_path.exists() and not replace:
                raise CoinbaseAccountError("already_configured_use_replace")
            credentials = CoinbaseAccountCredentials.from_payload(
                _read_json(key_file, downloaded=True)
            )
        else:
            connection = _read_json(connection_path)
            credentials = CoinbaseAccountCredentials.from_payload(
                connection.get("credentials")
            )
        client = client_factory(credentials)
        try:
            snapshot = client.snapshot()
        finally:
            client.close()
        verified_at = _now()
        # Credentials and the matching complete snapshot publish as one generation.
        _write_json(
            connection_path,
            {
                "schema_version": 1,
                "verified_at": verified_at,
                "credentials": credentials.payload(),
                "snapshot": snapshot,
            },
        )
        result.update(
            overall_status="ready",
            connected=True,
            verified_at=verified_at,
            account_count=len(snapshot["accounts"]),
            permissions=snapshot["permissions"],
            scope=snapshot["scope"],
        )
    except CoinbaseAccountError as exc:
        result["reason"] = str(exc)
    except BlockingIOError:
        result["reason"] = "account_sync_already_running"
    except Exception:
        result["reason"] = "account_sync_failed"
    finally:
        if locked:
            try:
                _write_json(
                    state_dir / "status.json",
                    {**result, "verified_at": result.get("verified_at", _now())},
                )
            except Exception:
                result.update(
                    overall_status="blocked",
                    connected=False,
                    reason="status_write_failed",
                )
        if lock_fd is not None:
            os.close(lock_fd)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--link-key-file", type=Path, help="Path only; never pass secret contents."
    )
    group.add_argument(
        "--status", action="store_true", help="Local status only; no API requests."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace a previously linked key after verification.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.replace and args.link_key_file is None:
        parser.error("--replace requires --link-key-file")
    payload = (
        status()
        if args.status
        else sync_account(
            key_file=args.link_key_file.expanduser() if args.link_key_file else None,
            replace=args.replace,
        )
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"coinbase_account status={payload['overall_status']} mode=read_only "
            f"account_count={payload['account_count']} reason={payload.get('reason', '')}"
        )
    return 0 if payload["overall_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
