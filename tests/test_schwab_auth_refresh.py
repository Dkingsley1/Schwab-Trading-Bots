import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

from scripts.ops import schwab_auth_refresh as sar

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_script_path_execution_bootstraps_project_imports(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    out_path = tmp_path / "auth.json"
    env = os.environ.copy()
    env.pop("SCHWAB_API_KEY", None)
    env.pop("SCHWAB_SECRET", None)
    env.pop("PYTHONPATH", None)

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ops" / "schwab_auth_refresh.py"),
            "--token-path",
            str(token_path),
            "--out-file",
            str(out_path),
            "--json",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "ModuleNotFoundError" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["reason"] == "missing_credentials"


def test_token_status_reads_nested_epoch_expiry() -> None:
    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / "token.json"
        token_path.write_text(
            json.dumps(
                {
                    "creation_timestamp": 1773069288,
                    "token": {
                        "refresh_token": "refresh-token",
                        "access_token": "access-token",
                        "expires_at": 4102444800,
                    },
                }
            ),
            encoding="utf-8",
        )

        status = sar._token_status(token_path)

    assert status["exists"] is True
    assert status["expires_at"] == "4102444800"
    assert float(status["expires_in_seconds"]) > 0.0


def test_token_needs_refresh_uses_min_expiry_floor() -> None:
    status = {
        "exists": True,
        "size_bytes": 808,
        "expires_in_seconds": 450.0,
    }

    needs_refresh, reason = sar._token_needs_refresh(status, min_expires_seconds=300.0)
    assert needs_refresh is False
    assert reason == "token_ready"

    needs_refresh, reason = sar._token_needs_refresh(status, min_expires_seconds=600.0)
    assert needs_refresh is True
    assert reason.startswith("token_expiring_soon:")


def test_normalize_browser_app_name_maps_common_aliases() -> None:
    assert sar._normalize_browser_app_name("chrome") == "Google Chrome"
    assert sar._normalize_browser_app_name("safari") == "Safari"
    assert sar._normalize_browser_app_name("msedge") == "Microsoft Edge"
    assert sar._normalize_browser_app_name("") is None


def test_browser_disabled_blocks_interactive_refresh_before_opening_browser(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    out_path = tmp_path / "auth.json"
    env = os.environ.copy()
    env.update(
        {
            "SCHWAB_API_KEY": "test_key",
            "SCHWAB_SECRET": "test_secret",
            "SCHWAB_REDIRECT": "https://127.0.0.1:8182",
            "SCHWAB_AUTH_BROWSER_DISABLED": "1",
            "SCHWAB_AUTH_ALLOW_BROWSER_OPEN": "0",
        }
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ops" / "schwab_auth_refresh.py"),
            "--token-path",
            str(token_path),
            "--out-file",
            str(out_path),
            "--json",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 3
    payload = json.loads(proc.stdout)
    assert payload["reason"] == "browser_disabled_token_refresh_required"
    assert payload["refresh_needed_after"] is True


def test_open_url_via_applescript_activates_requested_browser(monkeypatch) -> None:
    seen = []

    def fake_run(command, capture_output, text, check):
        seen.append(command)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sar.sys, "platform", "darwin", raising=False)
    monkeypatch.setattr(sar.subprocess, "run", fake_run)

    ok, method = sar._open_url_via_applescript("https://example.com", "chrome")

    assert ok is True
    assert method == "applescript_app:Google Chrome"
    assert seen == [[
        "/usr/bin/osascript",
        "-e",
        'tell application "Google Chrome"\nactivate\nopen location "https://example.com"\nend tell\n',
    ]]


def test_open_url_via_macos_uses_requested_browser_alias(monkeypatch) -> None:
    seen = []

    def fake_run(command, capture_output, text, check):
        seen.append(command)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sar.sys, "platform", "darwin", raising=False)
    monkeypatch.setattr(sar.subprocess, "run", fake_run)

    ok, method = sar._open_url_via_macos("https://example.com", "chrome")

    assert ok is True
    assert method == "applescript_app:Google Chrome"
    assert seen == [[
        "/usr/bin/osascript",
        "-e",
        'tell application "Google Chrome"\nactivate\nopen location "https://example.com"\nend tell\n',
    ]]


def test_open_url_via_macos_falls_back_to_default_browser(monkeypatch) -> None:
    seen = []
    responses = iter(
        [
            types.SimpleNamespace(returncode=1, stdout="", stderr="applescript_failed"),
            types.SimpleNamespace(returncode=1, stdout="", stderr="app_missing"),
            types.SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    def fake_run(command, capture_output, text, check):
        seen.append(command)
        return next(responses)

    monkeypatch.setattr(sar.sys, "platform", "darwin", raising=False)
    monkeypatch.setattr(sar.subprocess, "run", fake_run)

    ok, method = sar._open_url_via_macos("https://example.com", "chrome")

    assert ok is True
    assert method == "open_default"
    assert seen == [
        ["/usr/bin/osascript", "-e", 'tell application "Google Chrome"\nactivate\nopen location "https://example.com"\nend tell\n'],
        ["/usr/bin/open", "-a", "Google Chrome", "https://example.com"],
        ["/usr/bin/open", "https://example.com"],
    ]


def test_main_opens_browser_without_extra_prompt_by_default(monkeypatch) -> None:
    seen = {}
    fake_module = types.ModuleType("core.base_trader")
    install_calls = {"count": 0}

    class FakeBaseTrader:
        def __init__(self, *args, **kwargs) -> None:
            self.client = None
            self.token_path = ""

        def authenticate(self):
            seen["interactive"] = os.environ["SCHWAB_AUTH_INTERACTIVE"]
            Path(self.token_path).write_text(
                json.dumps(
                    {
                        "token": {
                            "access_token": "access-token",
                            "refresh_token": "refresh-token",
                            "expires_at": 4102444800,
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.client = object()
            return self.client

    fake_module.BaseTrader = FakeBaseTrader
    monkeypatch.setitem(sys.modules, "core.base_trader", fake_module)
    monkeypatch.setenv("SCHWAB_API_KEY", "real_key")
    monkeypatch.setenv("SCHWAB_SECRET", "real_secret")
    monkeypatch.setattr(sar, "_install_schwab_browser_fallback", lambda: install_calls.__setitem__("count", install_calls["count"] + 1) or True)

    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / "token.json"
        out_path = Path(td) / "auth.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "schwab_auth_refresh.py",
                "--token-path",
                str(token_path),
                "--out-file",
                str(out_path),
                "--skip-account-probe",
            ],
        )

        assert sar.main() == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert seen["interactive"] == "0"
    assert payload["prompt_before_browser"] is False
    assert payload["requested_browser"] == "chrome"
    assert payload["requested_browser_resolved"] == "Google Chrome"
    assert install_calls["count"] == 1


def test_main_skips_browser_when_token_already_ready(monkeypatch) -> None:
    fake_module = types.ModuleType("core.base_trader")
    install_calls = {"count": 0}

    class FakeBaseTrader:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("ready token should not open Schwab auth flow")

    fake_module.BaseTrader = FakeBaseTrader
    monkeypatch.setitem(sys.modules, "core.base_trader", fake_module)
    monkeypatch.setenv("SCHWAB_API_KEY", "real_key")
    monkeypatch.setenv("SCHWAB_SECRET", "real_secret")
    monkeypatch.setattr(sar, "_install_schwab_browser_fallback", lambda: install_calls.__setitem__("count", install_calls["count"] + 1) or True)

    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / "token.json"
        token_path.write_text(
            json.dumps(
                {
                    "token": {
                        "access_token": "access-token",
                        "refresh_token": "refresh-token",
                        "expires_at": 4102444800,
                    }
                }
            ),
            encoding="utf-8",
        )
        out_path = Path(td) / "auth.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "schwab_auth_refresh.py",
                "--token-path",
                str(token_path),
                "--out-file",
                str(out_path),
                "--json",
            ],
        )

        assert sar.main() == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert payload["ok"] is True
    assert payload["skipped"] is True
    assert payload["reason"] == "token_already_ready"
    assert payload["refresh_needed_before"] is False
    assert install_calls["count"] == 0


def test_main_can_restore_enter_gate_before_browser(monkeypatch) -> None:
    seen = {}
    fake_module = types.ModuleType("core.base_trader")
    install_calls = {"count": 0}

    class FakeBaseTrader:
        def __init__(self, *args, **kwargs) -> None:
            self.client = None
            self.token_path = ""

        def authenticate(self):
            seen["interactive"] = os.environ["SCHWAB_AUTH_INTERACTIVE"]
            Path(self.token_path).write_text(
                json.dumps(
                    {
                        "token": {
                            "access_token": "access-token",
                            "refresh_token": "refresh-token",
                            "expires_at": 4102444800,
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.client = object()
            return self.client

    fake_module.BaseTrader = FakeBaseTrader
    monkeypatch.setitem(sys.modules, "core.base_trader", fake_module)
    monkeypatch.setenv("SCHWAB_API_KEY", "real_key")
    monkeypatch.setenv("SCHWAB_SECRET", "real_secret")
    monkeypatch.setattr(sar, "_install_schwab_browser_fallback", lambda: install_calls.__setitem__("count", install_calls["count"] + 1) or True)

    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / "token.json"
        out_path = Path(td) / "auth.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "schwab_auth_refresh.py",
                "--token-path",
                str(token_path),
                "--out-file",
                str(out_path),
                "--skip-account-probe",
                "--prompt-before-browser",
            ],
        )

        assert sar.main() == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert seen["interactive"] == "1"
    assert payload["prompt_before_browser"] is True
    assert install_calls["count"] == 1


def test_successful_interactive_account_probe_triggers_post_refresh_cascade(monkeypatch) -> None:
    fake_module = types.ModuleType("core.base_trader")
    cascade_calls = {"count": 0}

    class FakeResponse:
        status_code = 200

    class FakeBaseTrader:
        def __init__(self, *args, **kwargs) -> None:
            self.client = self
            self.token_path = ""

        def authenticate(self):
            Path(self.token_path).write_text(
                json.dumps(
                    {
                        "token": {
                            "access_token": "access-token",
                            "refresh_token": "refresh-token",
                            "expires_at": 4102444800,
                        }
                    }
                ),
                encoding="utf-8",
            )
            return self

        def get_account_numbers(self):
            return FakeResponse()

    fake_module.BaseTrader = FakeBaseTrader
    monkeypatch.setitem(sys.modules, "core.base_trader", fake_module)
    monkeypatch.setenv("SCHWAB_API_KEY", "real_key")
    monkeypatch.setenv("SCHWAB_SECRET", "real_secret")
    monkeypatch.setattr(sar, "_install_schwab_browser_fallback", lambda: True)

    def fake_cascade(**kwargs):
        cascade_calls["count"] += 1
        return {
            "attempted": True,
            "ok": True,
            "overall_status": "ready",
            "refresh_completed": True,
            "paper_truth_ready": True,
            "steps": [],
        }

    monkeypatch.setattr(sar, "_run_post_refresh_cascade", fake_cascade)

    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / "token.json"
        out_path = Path(td) / "auth.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "schwab_auth_refresh.py",
                "--token-path",
                str(token_path),
                "--out-file",
                str(out_path),
                "--json",
            ],
        )

        assert sar.main() == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert cascade_calls["count"] == 1
    assert payload["overall_status"] == "ready"
    assert payload["post_refresh_cascade"]["paper_truth_ready"] is True
