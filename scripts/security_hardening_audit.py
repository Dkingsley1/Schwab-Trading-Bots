import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Security/production hygiene audit.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "health" / "security_audit_latest.json"))
    parser.add_argument("--secret-scan-max-age-hours", type=float, default=36.0)
    parser.add_argument("--audit-journal-max-age-hours", type=float, default=168.0)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    checks = []

    pre_commit = PROJECT_ROOT / ".githooks" / "pre-commit"
    checks.append({"name": "pre_commit_hook_exists", "ok": pre_commit.exists()})

    hook_text = pre_commit.read_text(encoding="utf-8") if pre_commit.exists() else ""
    checks.append({"name": "pre_commit_secret_scan_enabled", "ok": "secret_scan.py --staged" in hook_text})

    gitignore_text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8") if (PROJECT_ROOT / ".gitignore").exists() else ""
    checks.append({"name": "token_json_ignored", "ok": "token.json" in gitignore_text})

    approval = PROJECT_ROOT / "governance" / "champion_challenger" / "PROMOTION_APPROVED.flag"
    approval_ok = False
    if approval.exists():
        try:
            obj = json.loads(approval.read_text(encoding="utf-8"))
            approval_ok = bool(obj.get("approved_by")) and bool(obj.get("approved_at_utc")) and bool(obj.get("ticket"))
        except Exception:
            approval_ok = False
    checks.append({"name": "promotion_approval_signed_json", "ok": approval_ok or not approval.exists()})

    backup_dir = PROJECT_ROOT / "exports" / "env_snapshots"
    checks.append({"name": "backup_snapshot_exists", "ok": backup_dir.exists() and any(backup_dir.iterdir())})

    rbac_path = _first_existing(
        [
            PROJECT_ROOT / "config" / "security" / "rbac_roles.json",
            PROJECT_ROOT / "governance" / "security" / "rbac_roles.json",
        ]
    )
    rbac = _load_json(rbac_path)
    role_rows = rbac.get("roles") if isinstance(rbac.get("roles"), list) else []
    separation = rbac.get("separation_of_duties") if isinstance(rbac.get("separation_of_duties"), dict) else {}
    role_names = {str((row or {}).get("role") or "").strip() for row in role_rows if isinstance(row, dict)}
    required_roles = {
        "research_reviewer",
        "risk_reviewer",
        "live_operator",
        "risk_operator",
        "storage_maintainer",
        "audit_reviewer",
    }
    separation_ok = bool(
        separation.get("promotion_approval_requires_distinct_roles")
        and separation.get("live_execution_enable_requires_roles")
        and separation.get("artifact_delete_requires_roles")
    )
    checks.append({"name": "rbac_manifest_exists", "ok": bool(rbac)})
    checks.append({"name": "rbac_required_roles_present", "ok": required_roles.issubset(role_names)})
    checks.append({"name": "separation_of_duties_defined", "ok": separation_ok})

    key_rotation_path = _first_existing(
        [
            PROJECT_ROOT / "config" / "security" / "key_rotation_policy.json",
            PROJECT_ROOT / "governance" / "security" / "key_rotation_policy.json",
        ]
    )
    key_rotation = _load_json(key_rotation_path)
    rotation = key_rotation.get("rotation") if isinstance(key_rotation.get("rotation"), dict) else {}
    key_rotation_ok = bool(
        rotation
        and int(rotation.get("api_keys_days", 0) or 0) > 0
        and int(rotation.get("broker_tokens_days", 0) or 0) > 0
        and int(rotation.get("signing_keys_days", 0) or 0) > 0
    )
    checks.append({"name": "key_rotation_policy_exists", "ok": bool(key_rotation)})
    checks.append({"name": "key_rotation_schedule_defined", "ok": key_rotation_ok})

    secret_scan_path = PROJECT_ROOT / "governance" / "health" / "secret_scan_latest.json"
    secret_scan = _load_json(secret_scan_path)
    secret_scan_ts = _parse_iso_utc(secret_scan.get("timestamp_utc"))
    secret_scan_age_hours = (
        max((now - secret_scan_ts).total_seconds() / 3600.0, 0.0)
        if secret_scan_ts is not None
        else None
    )
    checks.append({"name": "secret_scan_artifact_present", "ok": bool(secret_scan)})
    checks.append(
        {
            "name": "secret_scan_artifact_fresh",
            "ok": secret_scan_age_hours is not None and secret_scan_age_hours <= float(args.secret_scan_max_age_hours),
        }
    )
    checks.append({"name": "secret_scan_clear", "ok": int(secret_scan.get("findings_count", 0) or 0) == 0})

    mutation_latest_path = PROJECT_ROOT / "governance" / "audits" / "registry_mutation_latest.json"
    mutation_latest = _load_json(mutation_latest_path)
    mutation_latest_ts = _parse_iso_utc(mutation_latest.get("timestamp_utc"))
    if mutation_latest_ts is None and mutation_latest_path.exists():
        mutation_latest_ts = datetime.fromtimestamp(mutation_latest_path.stat().st_mtime, tz=timezone.utc)
    mutation_latest_age_hours = (
        max((now - mutation_latest_ts).total_seconds() / 3600.0, 0.0)
        if mutation_latest_ts is not None
        else None
    )
    checks.append({"name": "mutation_latest_present", "ok": bool(mutation_latest) or mutation_latest_path.exists()})
    checks.append(
        {
            "name": "mutation_latest_fresh",
            "ok": mutation_latest_age_hours is not None and mutation_latest_age_hours <= float(args.audit_journal_max_age_hours),
        }
    )
    mutation_journal_matches = sorted(PROJECT_ROOT.glob("governance/audits/registry_mutation_journal_*.jsonl*"))
    checks.append({"name": "mutation_journal_present", "ok": bool(mutation_journal_matches)})

    shadow_preflight = PROJECT_ROOT / "scripts" / "shadow_preflight.py"
    checks.append({"name": "paper_live_separation_guard_exists", "ok": shadow_preflight.exists()})

    out = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "overall_status": "ready" if all(c["ok"] for c in checks) else "needs_work",
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "summary": {
            "passed_checks": sum(1 for row in checks if row["ok"]),
            "failed_checks": sum(1 for row in checks if not row["ok"]),
            "secret_scan_age_hours": round(secret_scan_age_hours, 3) if secret_scan_age_hours is not None else None,
            "secret_scan_findings_count": int(secret_scan.get("findings_count", 0) or 0),
            "rbac_role_count": len(role_names),
            "rbac_manifest_path": str(rbac_path),
            "key_rotation_policy_path": str(key_rotation_path),
            "key_rotation_schedule_defined": key_rotation_ok,
            "mutation_latest_age_hours": round(mutation_latest_age_hours, 3) if mutation_latest_age_hours is not None else None,
            "mutation_journal_files": len(mutation_journal_matches),
        },
        "recommendations": [
            "Keep secret-scan artifacts fresh and zero-finding before promotion or live enablement.",
            "Require RBAC, separation-of-duties policy, and paper/live preflight checks before expanding live permissions.",
            "Keep registry mutation journals fresh and define key-rotation cadences so governance evidence does not go stale between incidents.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
