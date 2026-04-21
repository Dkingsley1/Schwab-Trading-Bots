import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.security_hardening_audit as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_security_hardening_audit_checks_rbac_and_secret_scan(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".githooks").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".githooks" / "pre-commit").write_text("python scripts/secret_scan.py --staged\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("token.json\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "shadow_preflight.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (tmp_path / "exports" / "env_snapshots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "exports" / "env_snapshots" / "snapshot.json").write_text("{}", encoding="utf-8")
    (tmp_path / "governance" / "audits").mkdir(parents=True, exist_ok=True)
    (tmp_path / "governance" / "audits" / "registry_mutation_journal_20260406.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(
        tmp_path / "governance" / "security" / "rbac_roles.json",
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
        tmp_path / "governance" / "health" / "secret_scan_latest.json",
        {"timestamp_utc": "2026-04-06T12:00:00+00:00", "findings_count": 0},
    )

    monkeypatch.setattr(src, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["security_hardening_audit.py", "--out", str(tmp_path / "governance" / "health" / "security_audit_latest.json")])

    rc = src.main()
    payload = json.loads((tmp_path / "governance" / "health" / "security_audit_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["ok"] is True
    assert payload["summary"]["rbac_role_count"] == 6

