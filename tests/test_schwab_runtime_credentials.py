from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from core.broker_auth_epoch import token_epoch, token_epoch_changed
from core.brokers import schwab as schwab_broker
from core.brokers.models import BrokerAuthRequest, BrokerCredentials
from core.brokers.schwab_credentials import (
    DEFAULT_SERVICES,
    enforce_managed_schwab_runtime,
    managed_schwab_runtime_status,
    resolve_schwab_credentials,
    schwab_credentials_ready,
)
from scripts import run_all_sleeves as launcher


def test_keychain_runtime_resolution_without_secret_environment() -> None:
    values = {
        "SCHWAB_KEYCHAIN_RUNTIME_RESOLUTION_ENABLED": "1",
        "SCHWAB_KEYCHAIN_ACCOUNT": "test-account",
    }
    secrets = {
        DEFAULT_SERVICES["api_key"]: "key-from-keychain",
        DEFAULT_SERVICES["app_secret"]: "secret-from-keychain",
        DEFAULT_SERVICES["callback_url"]: "https://127.0.0.1:8182",
    }

    credentials = resolve_schwab_credentials(
        values,
        read_keychain=lambda service, account: secrets.get(service, "") if account == "test-account" else "",
    )

    assert schwab_credentials_ready(credentials) is True
    assert credentials.api_key == "key-from-keychain"
    assert credentials.app_secret == "secret-from-keychain"
    assert credentials.callback_url == "https://127.0.0.1:8182"


def test_keychain_runtime_resolution_is_explicit() -> None:
    calls: list[tuple[str, str]] = []

    credentials = resolve_schwab_credentials(
        {},
        read_keychain=lambda service, account: calls.append((service, account)) or "unexpected",
    )

    assert schwab_credentials_ready(credentials) is False
    assert calls == []


def test_token_epoch_changes_without_including_token_material(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    token_path.write_text(
        '{"access_token":"do-not-report","refresh_token":"also-private","expires_at":9999999999}',
        encoding="utf-8",
    )
    before = token_epoch(token_path)
    updated_ns = int(before["mtime_ns"]) + 1_000_000_000
    os.utime(token_path, ns=(updated_ns, updated_ns))
    after = token_epoch(token_path)

    assert token_epoch_changed(before, after) is True
    assert "access_token" not in str(after)
    assert "refresh_token" not in str(after)
    assert after["expires_at_epoch"] == 9999999999.0


def test_breaker_allows_fresh_measurement_during_bounded_auth_rotation_grace() -> None:
    now = 2_000.0
    metrics: dict[str, object] = {
        "_breaker_source_present": True,
        "_breaker_source_age_seconds": 5.0,
        "_broker_auth_epoch": {"present": True, "mtime_epoch": 1_900.0},
        "data_quality_session_local_timestamp": datetime.fromtimestamp(1_800.0, timezone.utc).isoformat(),
    }
    args = argparse.Namespace(
        breaker_max_metric_age_seconds=900.0,
        breaker_post_auth_evidence_grace_seconds=120.0,
        broker="schwab",
    )

    actionable, reason = launcher._breaker_metrics_actionable(metrics, args, now_epoch=now)

    assert actionable is True
    assert reason == "fresh_actionable_metrics"


def test_breaker_rejects_measurement_that_misses_auth_rotation_grace() -> None:
    now = 2_000.0
    metrics: dict[str, object] = {
        "_breaker_source_present": True,
        "_breaker_source_age_seconds": 5.0,
        "_broker_auth_epoch": {"present": True, "mtime_epoch": 1_950.0},
        "data_quality_session_local_timestamp": datetime.fromtimestamp(
            1_800.0, timezone.utc
        ).isoformat(),
    }
    args = argparse.Namespace(
        breaker_max_metric_age_seconds=900.0,
        breaker_post_auth_evidence_grace_seconds=120.0,
        broker="schwab",
    )

    actionable, reason = launcher._breaker_metrics_actionable(metrics, args, now_epoch=now)

    assert actionable is False
    assert reason == "measurement_predates_auth_epoch"


def test_latched_breaker_keeps_execution_job_parked_after_timer() -> None:
    spec = launcher.JobSpec(
        name="paper_executor",
        cmd=["python", "executor.py"],
        env={},
        breaker_group="execution",
    )

    parked = launcher._breaker_policy_parked_jobs(
        {spec.name: spec},
        {"execution": 0.0},
        now=1_000.0,
        latched_groups={"execution"},
    )

    assert parked == {"paper_executor"}


def test_auth_token_update_uses_in_process_rebind_without_launcher_recycle(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    token_path.write_text("{}", encoding="utf-8")
    os.utime(token_path, (500.0, 500.0))
    spec = launcher.JobSpec(
        name="schwab_collection",
        cmd=["python", "collector.py", "--broker", "schwab"],
        env={},
        breaker_group="collection",
        auth_watch_paths=(token_path,),
    )

    recycle, reason = launcher._job_recycle_due(spec, started_at=400.0, now_ts=600.0)

    assert spec.auth_change_mode == "in_process_rebind"
    assert recycle is False
    assert reason == ""


def test_schwab_adapter_applies_bounded_http_timeout(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.timeout = None

        def set_timeout(self, timeout: float) -> None:
            self.timeout = timeout

    client = FakeClient()
    monkeypatch.setenv("SCHWAB_API_TIMEOUT_SECONDS", "12")
    monkeypatch.setattr(schwab_broker, "_schwab_easy_client", lambda: lambda **kwargs: client)
    request = BrokerAuthRequest(
        credentials=BrokerCredentials("key", "secret", "https://127.0.0.1:8182"),
        token_path="token.json",
        max_token_age=None,
        callback_timeout=30.0,
        interactive=False,
        requested_browser=None,
    )

    result = schwab_broker.SchwabBrokerAdapter().authenticate(request)

    assert result is client
    assert client.timeout == 12.0


def test_managed_runtime_attestation_is_fail_closed_when_required() -> None:
    status = managed_schwab_runtime_status(
        {"SCHWAB_MANAGED_RUNTIME_REQUIRED": "1"}
    )

    assert status["ready"] is False
    try:
        enforce_managed_schwab_runtime(
            {"SCHWAB_MANAGED_RUNTIME_REQUIRED": "1"}
        )
    except RuntimeError as exc:
        assert str(exc) == "schwab_managed_runtime_attestation_required"
    else:
        raise AssertionError("missing managed runtime attestation must be rejected")


def test_managed_runtime_attestation_accepts_loader_source() -> None:
    status = enforce_managed_schwab_runtime(
        {
            "SCHWAB_MANAGED_RUNTIME_REQUIRED": "1",
            "SCHWAB_MANAGED_RUNTIME_ATTESTED": "1",
            "SCHWAB_MANAGED_RUNTIME_SOURCE": "load_runtime_env",
        }
    )

    assert status["ready"] is True
    assert status["secret_material_present"] is False
