#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import ops_data_plane

DEFAULT_CACHE_ROOT = PROJECT_ROOT / "governance" / "collector_cache"
_FILE_HASH_CHUNK_BYTES = 1024 * 1024


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_FILE_HASH_CHUNK_BYTES), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_timestamp(path: Path) -> float:
    payload = _load_json(path)
    for key in ("timestamp_utc", "updated_at_utc", "updated_at", "generated_utc", "created_at"):
        raw = str(payload.get(key) or "").strip()
        if not raw:
            continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        except Exception:
            continue
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _oldest_expected_timestamp(paths: list[Path]) -> float:
    values = [_payload_timestamp(path) for path in paths if path.exists()]
    return min(values) if values else 0.0


def _build_fingerprint(command: list[str], fingerprint_files: list[Path], expect_paths: list[Path]) -> str:
    payload = {
        "command": command,
        "fingerprint_files": [
            {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for path in fingerprint_files
        ],
        "expect_paths": [str(path) for path in expect_paths],
    }
    return _sha256_text(json.dumps(payload, ensure_ascii=True, sort_keys=True))


def _record_provenance(
    *,
    collector_key: str,
    cache_key: str,
    command: list[str],
    expect_paths: list[Path],
    fingerprint_files: list[Path],
    command_fingerprint: str,
    payload: dict[str, Any],
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> str:
    with ops_data_plane.connect(PROJECT_ROOT) as conn:
        run_uid = ops_data_plane.record_collector_run(
            conn,
            collector_key=collector_key,
            cache_key=cache_key,
            command=command,
            expect_paths=[str(path) for path in expect_paths],
            fingerprint_files=[str(path) for path in fingerprint_files],
            command_fingerprint=command_fingerprint,
            skipped=bool(payload.get("skipped", False)),
            rc=int(payload.get("rc", 0) or 0),
            started_utc=str(payload.get("started_utc") or payload.get("timestamp_utc") or _now_utc()),
            finished_utc=str(payload.get("finished_utc") or payload.get("timestamp_utc") or _now_utc()),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            payload_sha256=_sha256_text(json.dumps(payload, ensure_ascii=True, sort_keys=True)),
            metadata=payload,
            commit=False,
        )
        for expect_path in expect_paths:
            if not expect_path.exists():
                continue
            watermark_payload = _load_json(expect_path)
            artifact_key = ops_data_plane.normalize_entity_key(PROJECT_ROOT, expect_path)
            watermark_value = str(
                watermark_payload.get("timestamp_utc")
                or watermark_payload.get("generated_utc")
                or watermark_payload.get("updated_at_utc")
                or watermark_payload.get("updated_at")
                or datetime.fromtimestamp(expect_path.stat().st_mtime, tz=timezone.utc).isoformat()
            )
            ops_data_plane.record_watermark(
                conn,
                collector_key=collector_key,
                source_name=artifact_key,
                entity_key=artifact_key,
                watermark_type="artifact_timestamp",
                watermark_value=watermark_value,
                payload_sha256=_sha256_file(expect_path),
                metadata={
                    "collector_key": collector_key,
                    "cache_key": cache_key,
                    "run_uid": run_uid,
                    "artifact_key": artifact_key,
                    "artifact_path": str(expect_path),
                    "artifact_present": True,
                },
                commit=False,
            )
        conn.commit()
    return run_uid


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a collector only when its expected artifacts are stale.")
    parser.add_argument("--key", required=True)
    parser.add_argument("--max-age-minutes", type=float, required=True)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--expect-path", action="append", default=[])
    parser.add_argument("--fingerprint-file", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("run_cached_collector requires a command after --")

    expect_paths = [Path(p).expanduser() for p in args.expect_path if str(p).strip()]
    fingerprint_files = [Path(p).expanduser() for p in args.fingerprint_file if str(p).strip()]
    cache_root = Path(args.cache_root).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"{str(args.key).strip() or 'collector'}.json"
    state = _load_json(cache_file)
    fingerprint = _build_fingerprint(command, fingerprint_files, expect_paths)
    started_utc = _now_utc()

    max_age_seconds = max(float(args.max_age_minutes), 0.0) * 60.0
    all_expected_present = bool(expect_paths) and all(path.exists() for path in expect_paths)
    oldest_ts = _oldest_expected_timestamp(expect_paths) if all_expected_present else 0.0
    age_seconds = max(datetime.now(timezone.utc).timestamp() - oldest_ts, 0.0) if oldest_ts > 0 else float("inf")
    fresh_enough = all_expected_present and age_seconds <= max_age_seconds
    fingerprint_match = (not state) or (str(state.get("fingerprint") or "") == fingerprint)
    skipped = bool(fresh_enough and fingerprint_match)

    payload: dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "started_utc": started_utc,
        "key": str(args.key),
        "command": command,
        "expect_paths": [str(path) for path in expect_paths],
        "fingerprint_files": [str(path) for path in fingerprint_files],
        "fingerprint": fingerprint,
        "max_age_minutes": float(args.max_age_minutes),
        "age_seconds": None if age_seconds == float("inf") else round(float(age_seconds), 3),
        "all_expected_present": bool(all_expected_present),
        "skipped": skipped,
        "ran": not skipped,
        "cache_file": str(cache_file),
    }

    if skipped:
        payload["rc"] = 0
        payload["reason"] = "fresh_artifacts_reused"
        payload["finished_utc"] = _now_utc()
        payload["run_uid"] = _record_provenance(
            collector_key=str(args.key),
            cache_key=str(args.key),
            command=command,
            expect_paths=expect_paths,
            fingerprint_files=fingerprint_files,
            command_fingerprint=fingerprint,
            payload=payload,
        )
        cache_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"cached_collector skip key={args.key} age_s={payload['age_seconds']} command={' '.join(command)}")
        return 0

    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    payload.update(
        {
            "rc": int(proc.returncode),
            "finished_utc": _now_utc(),
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-12:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-12:]),
            "expect_paths_after": [str(path) for path in expect_paths if path.exists()],
        }
    )
    payload["run_uid"] = _record_provenance(
        collector_key=str(args.key),
        cache_key=str(args.key),
        command=command,
        expect_paths=expect_paths,
        fingerprint_files=fingerprint_files,
        command_fingerprint=fingerprint,
        payload=payload,
        stdout_tail=payload["stdout_tail"],
        stderr_tail=payload["stderr_tail"],
    )
    cache_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
