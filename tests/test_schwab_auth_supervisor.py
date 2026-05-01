import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import schwab_auth_supervisor as supervisor


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _token(path: Path, *, expires_at: int = 4102444800) -> None:
    path.write_text(
        json.dumps(
            {
                "token": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": expires_at,
                }
            }
        ),
        encoding="utf-8",
    )


def test_schwab_auth_supervisor_ready_for_fresh_token(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    token_path = project_root / "token.json"
    project_root.mkdir(parents=True)
    _token(token_path)
    _write_json(health / "premarket_token_guard_latest.json", {"ok": True})
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True, "auth_ok": True, "network_ok": True})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy"})

    monkeypatch.setattr(supervisor, "_list_auth_processes", lambda: [])
    monkeypatch.setattr(supervisor, "_callback_port_open", lambda *args, **kwargs: False)
    monkeypatch.setattr(supervisor, "_recent_auth_signals", lambda root: {"auth_error_markers": [], "callback_error_markers": [], "auth_error_count": 0, "callback_error_count": 0, "circuit_breaker_with_auth_error": False})

    payload = supervisor.build_payload(project_root, token_path=token_path)

    assert payload["overall_status"] == "ready"
    assert payload["token"]["ready"] is True
    assert payload["regression_contract"]["do_not_open_browser_when_token_ready"] is True


def test_schwab_auth_supervisor_degrades_and_cleans_stale_helpers(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    token_path = project_root / "token.json"
    project_root.mkdir(parents=True)
    _token(token_path)
    _write_json(health / "premarket_token_guard_latest.json", {"ok": True})
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True, "auth_ok": True, "network_ok": True})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy"})
    stale = supervisor.ProcessRow(pid=12345, ppid=1, elapsed_seconds=300, command="python scripts/ops/schwab_auth_refresh.py --json")
    killed: list[int] = []

    monkeypatch.setattr(supervisor, "_list_auth_processes", lambda: [stale])
    monkeypatch.setattr(supervisor, "_callback_port_open", lambda *args, **kwargs: True)
    monkeypatch.setattr(supervisor, "_recent_auth_signals", lambda root: {"auth_error_markers": [], "callback_error_markers": [], "auth_error_count": 0, "callback_error_count": 0, "circuit_breaker_with_auth_error": False})
    monkeypatch.setattr(supervisor, "_kill_process", lambda pid: killed.append(pid) or {"pid": pid, "ok": True, "signal": "TERM"})
    monkeypatch.setattr(supervisor, "_run_json", lambda cmd, **kwargs: {"cmd": cmd, "rc": 0, "timed_out": False, "payload": {"ok": True}})

    payload = supervisor.build_payload(project_root, apply=True, token_path=token_path)

    assert payload["overall_status"] == "degraded"
    assert "stale_schwab_auth_refresh_processes" in payload["findings"]
    assert "callback_port_held_by_stale_auth_helper" in payload["findings"]
    assert killed == [12345]
    assert any(row.get("action") == "kill_stale_auth_helper" for row in payload["attempts"])


def test_schwab_auth_supervisor_blocks_auth_errors_misclassified_as_symbol_failures(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    token_path = project_root / "token.json"
    project_root.mkdir(parents=True)
    _write_json(health / "premarket_token_guard_latest.json", {"ok": False})
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": False, "auth_ok": False, "network_ok": True})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "blocked", "lease_state": "critical"})

    monkeypatch.setattr(supervisor, "_list_auth_processes", lambda: [])
    monkeypatch.setattr(supervisor, "_callback_port_open", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        supervisor,
        "_recent_auth_signals",
        lambda root: {
            "auth_error_markers": ["refresh_token_authentication_error", "unsupported_token_type"],
            "callback_error_markers": [],
            "auth_error_count": 2,
            "callback_error_count": 0,
            "circuit_breaker_with_auth_error": True,
        },
    )

    payload = supervisor.build_payload(project_root, token_path=token_path)

    assert payload["overall_status"] == "blocked"
    assert "auth_error_misclassified_as_symbol_data" in payload["findings"]
    assert payload["operator_followups"] == ["./scripts/ops/opsctl.sh token-refresh-interactive --force --json"]
