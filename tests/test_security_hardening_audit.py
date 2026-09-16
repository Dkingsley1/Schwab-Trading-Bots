import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.security_hardening_audit as src
import pytest

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_security_hardening_audit_checks_rbac_and_secret_scan(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc).isoformat()
    (tmp_path / ".githooks").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".githooks" / "pre-commit").write_text("python scripts/secret_scan.py --staged\n", encoding="utf-8")
    (tmp_path / ".githooks" / "pre-commit").chmod(0o700)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "config", "--local", "core.hooksPath", ".githooks"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / ".gitignore").write_text("token.json\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "shadow_preflight.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (tmp_path / "exports" / "env_snapshots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "exports" / "env_snapshots" / "snapshot.json").write_text("{}", encoding="utf-8")
    (tmp_path / "governance" / "audits").mkdir(parents=True, exist_ok=True)
    (tmp_path / "governance" / "audits" / "registry_mutation_latest.json").write_text(
        json.dumps({"timestamp_utc": now}),
        encoding="utf-8",
    )
    (tmp_path / "governance" / "audits" / "registry_mutation_journal_20260406.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(
        tmp_path / "config" / "security" / "rbac_roles.json",
        {
            "roles": [
                {"role": "research_reviewer"},
                {"role": "risk_reviewer"},
                {"role": "live_operator"},
                {"role": "risk_operator"},
                {"role": "storage_maintainer"},
                {"role": "audit_reviewer"},
            ],
            "separation_of_duties": {
                "promotion_approval_requires_distinct_roles": ["research_reviewer", "risk_reviewer"],
                "live_execution_enable_requires_roles": ["live_operator", "risk_operator"],
                "artifact_delete_requires_roles": ["storage_maintainer", "audit_reviewer"],
            },
        },
    )
    _write_json(
        tmp_path / "config" / "security" / "key_rotation_policy.json",
        {
            "rotation": {
                "api_keys_days": 30,
                "broker_tokens_days": 7,
                "signing_keys_days": 90,
            }
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "secret_scan_latest.json",
        {"timestamp_utc": now, "findings_count": 0},
    )

    monkeypatch.setattr(src, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["security_hardening_audit.py", "--out", str(tmp_path / "governance" / "health" / "security_audit_latest.json")])

    rc = src.main()
    payload = json.loads((tmp_path / "governance" / "health" / "security_audit_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["ok"] is True
    assert payload["summary"]["rbac_role_count"] == 6
    assert payload["summary"]["key_rotation_schedule_defined"] is True
    assert payload["summary"]["rbac_manifest_path"].endswith("config/security/rbac_roles.json")


def test_checked_in_hook_without_git_activation_is_not_enabled(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "config", "--local", "core.hooksPath", ".git/hooks"],
        cwd=tmp_path,
        check=True,
    )
    hook = tmp_path / ".githooks/pre-commit"
    hook.parent.mkdir()
    hook.write_text("python scripts/secret_scan.py --staged\n")
    hook.chmod(0o700)
    assert src._hook_contract(tmp_path)["active"] is False


def test_nonexecutable_hook_does_not_count_as_active(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "config", "--local", "core.hooksPath", ".githooks"],
        cwd=tmp_path,
        check=True,
    )
    hook = tmp_path / ".githooks/pre-commit"
    hook.parent.mkdir()
    hook.write_text("python scripts/secret_scan.py --staged\n")
    hook.chmod(0o600)
    assert src._hook_contract(tmp_path)["active"] is False


@pytest.mark.parametrize("timestamp", [None, "invalid", "2099-01-01T00:00:00+00:00"])
def test_security_receipts_cannot_use_file_mtime_or_future_time(
    tmp_path, monkeypatch, timestamp
):
    row = (
        {"timestamp_utc": timestamp} if timestamp is not None else {"unversioned": True}
    )
    _write_json(tmp_path / "governance/health/secret_scan_latest.json", row)
    _write_json(tmp_path / "governance/audits/registry_mutation_latest.json", row)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(src, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["audit", "--out", str(output), "--json"])
    assert src.main() == 2
    checks = {
        row["name"]: row["ok"] for row in json.loads(output.read_text())["checks"]
    }
    assert checks["secret_scan_artifact_fresh"] is False
    assert checks["secret_scan_clear"] is False
    assert checks["mutation_latest_fresh"] is False
