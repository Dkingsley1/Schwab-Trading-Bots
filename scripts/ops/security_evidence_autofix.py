#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, write_payload
else:
    from core.runtime_python import resolve_runtime_python
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "security_evidence_autofix_latest.json"
SECRET_SCAN_PATH = PROJECT_ROOT / "governance" / "health" / "secret_scan_latest.json"
MUTATION_LATEST_PATH = PROJECT_ROOT / "governance" / "audits" / "registry_mutation_latest.json"

Runner = Callable[[list[str], Path, float | None], dict[str, Any]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _payload_age_hours(payload: dict[str, Any], path: Path) -> float | None:
    ts = parse_iso_utc(payload.get("timestamp_utc"))
    if ts is None and path.exists():
        try:
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            ts = None
    if ts is None:
        return None
    return max((datetime.now(timezone.utc) - ts).total_seconds() / 3600.0, 0.0)


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run_json(cmd: list[str], project_root: Path, timeout_sec: float | None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
        return {
            "cmd": list(cmd),
            "rc": int(proc.returncode),
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-12:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-12:]),
            "payload": _parse_json_output(proc.stdout or ""),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "cmd": list(cmd),
            "rc": 124,
            "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-12:]) or "timeout",
            "payload": _parse_json_output(stdout),
        }


def _latest_journal_entry(project_root: Path) -> tuple[Path | None, dict[str, Any]]:
    audit_root = project_root / "governance" / "audits"
    candidates = sorted(
        audit_root.glob("registry_mutation_journal_*.jsonl*"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                rows = [line.strip() for line in handle if line.strip()]
        except Exception:
            continue
        for raw in reversed(rows):
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                return path, payload
    return None, {}


def _mutation_latest_contract(path: Path, payload: dict[str, Any], *, journal_count: int) -> dict[str, Any]:
    age_hours = _payload_age_hours(payload, path)
    present = bool(payload) or path.exists()
    return {
        "present": present,
        "age_hours": round(age_hours, 3) if age_hours is not None else None,
        "journal_count": int(journal_count),
        "fresh": age_hours is not None and age_hours <= 168.0,
        "ok": present and bool(journal_count > 0),
    }


def _bootstrap_registry_mutation(project_root: Path) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    registry_payload = load_json(registry_path)
    rows = registry_payload.get("sub_bots") if isinstance(registry_payload.get("sub_bots"), list) else []
    registry_text = registry_path.read_text(encoding="utf-8") if registry_path.exists() else ""
    registry_sha = hashlib.sha256(registry_text.encode("utf-8")).hexdigest() if registry_text else ""
    return {
        "timestamp_utc": iso_now(),
        "actor": "security_evidence_autofix",
        "reason": "bootstrap_current_registry_baseline",
        "bootstrap": True,
        "mutation": {
            "bots_total_before": int(len(rows)),
            "bots_total_after": int(len(rows)),
            "bot_diff_count": 0,
            "bot_diffs": [],
            "registry_sha256_before": registry_sha,
            "registry_sha256_after": registry_sha,
        },
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply_repairs: bool = True,
    secret_scan_max_bytes: int = 1_000_000,
    secret_scan_max_age_hours: float = 36.0,
    mutation_max_age_hours: float = 168.0,
    force_secret_scan: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    run_json = runner or _run_json
    health_root = project_root / "governance" / "health"
    audits_root = project_root / "governance" / "audits"
    secret_scan_path = health_root / "secret_scan_latest.json"
    mutation_latest_path = audits_root / "registry_mutation_latest.json"

    secret_scan = load_json(secret_scan_path)
    secret_scan_age_hours = _payload_age_hours(secret_scan, secret_scan_path)
    secret_scan_stale = not secret_scan or secret_scan_age_hours is None or secret_scan_age_hours > float(secret_scan_max_age_hours)
    secret_scan_findings = int(secret_scan.get("findings_count", 0) or 0)

    secret_scan_refresh_needed = bool(secret_scan_stale or force_secret_scan)
    secret_scan_refresh = {
        "requested": bool(apply_repairs and secret_scan_refresh_needed),
        "applied": False,
        "cmd": [],
        "rc": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if apply_repairs and secret_scan_refresh_needed:
        result = run_json(
            [
                str(PY),
                str(project_root / "scripts" / "secret_scan.py"),
                "--max-bytes",
                str(max(int(secret_scan_max_bytes), 1)),
                "--out",
                str(secret_scan_path),
            ],
            project_root,
            900.0,
        )
        secret_scan = load_json(secret_scan_path)
        secret_scan_age_hours = _payload_age_hours(secret_scan, secret_scan_path)
        secret_scan_findings = int(secret_scan.get("findings_count", 0) or 0)
        secret_scan_refresh = {
            "requested": True,
            "applied": True,
            "cmd": list(result.get("cmd") or []),
            "rc": int(result.get("rc", 1)),
            "stdout_tail": str(result.get("stdout_tail") or ""),
            "stderr_tail": str(result.get("stderr_tail") or ""),
        }

    mutation_latest = load_json(mutation_latest_path)
    mutation_age_hours = _payload_age_hours(mutation_latest, mutation_latest_path)
    mutation_stale = not mutation_latest or mutation_age_hours is None or mutation_age_hours > float(mutation_max_age_hours)
    latest_journal_path, latest_journal_entry = _latest_journal_entry(project_root)
    journal_files = sorted(audits_root.glob("registry_mutation_journal_*.jsonl*"))
    mutation_refresh_applied = False
    mutation_refresh_reason = ""
    if apply_repairs and mutation_stale and latest_journal_entry:
        refreshed_payload = {
            **latest_journal_entry,
            "timestamp_utc": str(latest_journal_entry.get("timestamp_utc") or iso_now()),
            "refreshed_from_journal": True,
            "journal_source_path": str(latest_journal_path) if latest_journal_path is not None else "",
        }
        mutation_latest_path.parent.mkdir(parents=True, exist_ok=True)
        mutation_latest_path.write_text(json.dumps(refreshed_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        mutation_latest = refreshed_payload
        mutation_age_hours = _payload_age_hours(mutation_latest, mutation_latest_path)
        mutation_refresh_applied = True
        mutation_refresh_reason = "latest_refreshed_from_journal"
    elif apply_repairs and mutation_stale and not latest_journal_entry:
        bootstrap_payload = _bootstrap_registry_mutation(project_root)
        audits_root.mkdir(parents=True, exist_ok=True)
        journal_path = audits_root / f"registry_mutation_journal_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(bootstrap_payload, ensure_ascii=True) + "\n")
        mutation_latest_path.write_text(json.dumps(bootstrap_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        mutation_latest = bootstrap_payload
        mutation_age_hours = _payload_age_hours(mutation_latest, mutation_latest_path)
        latest_journal_path = journal_path
        latest_journal_entry = bootstrap_payload
        journal_files = sorted(audits_root.glob("registry_mutation_journal_*.jsonl*"))
        mutation_refresh_applied = True
        mutation_refresh_reason = "bootstrapped_current_registry_baseline"
    elif mutation_stale and not latest_journal_entry:
        mutation_refresh_reason = "journal_missing"

    secret_scan_ready = bool(secret_scan) and secret_scan_findings == 0 and secret_scan_age_hours is not None and secret_scan_age_hours <= float(secret_scan_max_age_hours)
    mutation_ready = bool(mutation_latest) and bool(journal_files) and mutation_age_hours is not None and mutation_age_hours <= float(mutation_max_age_hours)

    blockers = ordered_unique(
        [
            "secret_scan_missing_or_stale" if not bool(secret_scan) or secret_scan_age_hours is None or secret_scan_age_hours > float(secret_scan_max_age_hours) else "",
            "secret_scan_findings_present" if secret_scan_findings > 0 else "",
            "registry_mutation_latest_missing_or_stale" if not bool(mutation_latest) or mutation_age_hours is None or mutation_age_hours > float(mutation_max_age_hours) else "",
            "registry_mutation_journal_missing" if not journal_files else "",
        ]
    )
    overall_status = "ready"
    if blockers:
        overall_status = "blocked" if any("findings" in item or "journal_missing" in item for item in blockers) else "degraded"

    recommended_actions = ordered_unique(
        [
            "keep the full-repo secret scan fresh before promotion or live enablement" if not secret_scan_ready else "",
            "clear secret-scan findings before widening live permissions" if secret_scan_findings > 0 else "",
            "refresh registry mutation latest from the audit journal whenever the latest summary goes stale" if not mutation_ready and latest_journal_entry else "",
            "restore registry mutation journaling on registry writers so audit evidence cannot go dark" if not journal_files else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply_repairs": bool(apply_repairs),
        "force_secret_scan": bool(force_secret_scan),
        "secret_scan": {
            "path": str(secret_scan_path),
            "present": bool(secret_scan),
            "age_hours": round(secret_scan_age_hours, 3) if secret_scan_age_hours is not None else None,
            "fresh": secret_scan_age_hours is not None and secret_scan_age_hours <= float(secret_scan_max_age_hours),
            "findings_count": int(secret_scan_findings),
            "ok": secret_scan_ready,
        },
        "mutation_latest": {
            **_mutation_latest_contract(mutation_latest_path, mutation_latest, journal_count=len(journal_files)),
            "path": str(mutation_latest_path),
            "journal_source_path": str(latest_journal_path) if latest_journal_path is not None else "",
            "refreshed_from_journal": bool(mutation_refresh_applied),
            "refresh_reason": mutation_refresh_reason,
        },
        "repair_actions": {
            "secret_scan_refresh": secret_scan_refresh,
            "mutation_latest_refresh_applied": bool(mutation_refresh_applied),
        },
        "blockers": blockers,
        "recommended_actions": recommended_actions,
        "source_artifacts": {
            "secret_scan": str(secret_scan_path),
            "registry_mutation_latest": str(mutation_latest_path),
            "registry_mutation_journals": [str(path) for path in journal_files[:10]],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh security evidence surfaces like secret scans and mutation audit summaries.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--secret-scan-max-bytes", type=int, default=1_000_000)
    parser.add_argument("--secret-scan-max-age-hours", type=float, default=36.0)
    parser.add_argument("--mutation-max-age-hours", type=float, default=168.0)
    parser.add_argument("--force-secret-scan", action="store_true")
    parser.add_argument("--no-apply-repairs", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply_repairs=not bool(args.no_apply_repairs),
        secret_scan_max_bytes=int(args.secret_scan_max_bytes),
        secret_scan_max_age_hours=float(args.secret_scan_max_age_hours),
        mutation_max_age_hours=float(args.mutation_max_age_hours),
        force_secret_scan=bool(args.force_secret_scan),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "security_evidence_autofix "
            f"overall_status={payload.get('overall_status', '')} "
            f"secret_scan_ok={int(bool(((payload.get('secret_scan') or {}).get('ok', False))))} "
            f"mutation_ok={int(bool(((payload.get('mutation_latest') or {}).get('ok', False))))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
