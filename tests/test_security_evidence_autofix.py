import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import security_evidence_autofix as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_security_evidence_autofix_refreshes_mutation_latest_from_journal(tmp_path: Path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    audits = tmp_path / "governance" / "audits"
    now = datetime.now(timezone.utc).isoformat()
    _write_json(health / "secret_scan_latest.json", {"timestamp_utc": now, "findings_count": 0})
    audits.mkdir(parents=True, exist_ok=True)
    (audits / "registry_mutation_journal_20260422.jsonl").write_text(
        json.dumps({"timestamp_utc": now, "actor": "bot", "reason": "repair"}) + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(
        tmp_path,
        apply_repairs=True,
        runner=lambda cmd, project_root, timeout_sec: {"cmd": cmd, "rc": 0, "payload": {}, "stdout_tail": "", "stderr_tail": ""},
    )

    latest = json.loads((audits / "registry_mutation_latest.json").read_text(encoding="utf-8"))
    assert payload["mutation_latest"]["refreshed_from_journal"] is True
    assert latest["actor"] == "bot"
    assert latest["refreshed_from_journal"] is True


def test_security_evidence_autofix_blocks_when_secret_scan_has_findings(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    audits = tmp_path / "governance" / "audits"
    now = datetime.now(timezone.utc).isoformat()
    _write_json(health / "secret_scan_latest.json", {"timestamp_utc": now, "findings_count": 2})
    _write_json(audits / "registry_mutation_latest.json", {"timestamp_utc": now})
    (audits / "registry_mutation_journal_20260422.jsonl").write_text("{}\n", encoding="utf-8")

    payload = src.build_payload(tmp_path, apply_repairs=False)

    assert payload["overall_status"] == "blocked"
    assert "secret_scan_findings_present" in payload["blockers"]


def test_security_evidence_autofix_bootstraps_mutation_baseline_when_history_is_missing(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    registry = tmp_path / "master_bot_registry.json"
    now = datetime.now(timezone.utc).isoformat()
    _write_json(health / "secret_scan_latest.json", {"timestamp_utc": now, "findings_count": 0})
    _write_json(registry, {"sub_bots": [{"bot_id": "bot_a", "active": True}]})

    payload = src.build_payload(tmp_path, apply_repairs=True)

    latest_path = tmp_path / "governance" / "audits" / "registry_mutation_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert payload["mutation_latest"]["refreshed_from_journal"] is True
    assert payload["mutation_latest"]["refresh_reason"] == "bootstrapped_current_registry_baseline"
    assert latest["bootstrap"] is True
