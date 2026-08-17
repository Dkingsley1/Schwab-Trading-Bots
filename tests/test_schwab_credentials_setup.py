from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import schwab_credentials_setup as src


def test_schwab_credentials_setup_stores_to_keychain_without_logging_secret_values(monkeypatch, tmp_path: Path) -> None:
    store: dict[tuple[str, str], str] = {}
    prompts = iter(["api_key_value", "secret_value"])

    monkeypatch.setenv("SCHWAB_KEYCHAIN_ACCOUNT", "test-user")
    monkeypatch.delenv("SCHWAB_API_KEY", raising=False)
    monkeypatch.delenv("SCHWAB_SECRET", raising=False)
    monkeypatch.delenv("SCHWAB_REDIRECT", raising=False)
    monkeypatch.delenv("SCHWAB_CALLBACK_URL", raising=False)

    def read_keychain(service: str, account: str) -> str:
        return store.get((service, account), "")

    def write_keychain(service: str, account: str, value: str) -> tuple[bool, str]:
        store[(service, account)] = value
        return True, "stored"

    payload = src.build_payload(
        interactive=True,
        store="keychain",
        force=False,
        env_file=tmp_path / ".env.secrets.local",
        out_path=tmp_path / "schwab_credentials_setup_latest.json",
        input_fn=lambda prompt: "https://127.0.0.1:8182",
        secret_prompt_fn=lambda prompt: next(prompts),
        read_keychain=read_keychain,
        write_keychain=write_keychain,
    )
    text = json.dumps(payload)

    assert payload["overall_status"] == "ready"
    assert payload["credential_sources_after"]["keychain_present"]["api_key"] is True
    assert payload["credential_sources_after"]["keychain_present"]["secret"] is True
    assert payload["policy"]["headless_browser"] == "never"
    assert "api_key_value" not in text
    assert "secret_value" not in text


def test_schwab_credentials_setup_env_file_fallback_is_chmod_600(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / "config" / ".env.secrets.local"
    prompts = iter(["api_key_value", "secret_value"])
    monkeypatch.delenv("SCHWAB_API_KEY", raising=False)
    monkeypatch.delenv("SCHWAB_SECRET", raising=False)

    payload = src.build_payload(
        interactive=True,
        store="env-file",
        force=False,
        env_file=env_file,
        out_path=tmp_path / "schwab_credentials_setup_latest.json",
        input_fn=lambda prompt: "",
        secret_prompt_fn=lambda prompt: next(prompts),
        read_keychain=lambda service, account: "",
        write_keychain=lambda service, account, value: (False, "unexpected"),
    )
    mode = env_file.stat().st_mode & 0o777
    content = env_file.read_text(encoding="utf-8")

    assert payload["overall_status"] == "ready"
    assert payload["result"]["changed"] is True
    assert payload["credential_sources_after"]["env_file_present"]["api_key"] is True
    assert mode == 0o600
    assert "SCHWAB_API_KEY='api_key_value'" in content
    assert "SCHWAB_SECRET='secret_value'" in content
    assert "SCHWAB_CALLBACK_URL='https://127.0.0.1:8182'" in content
