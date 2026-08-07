import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import source_verification_autorefresh as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_runtime(project_root: Path, *, ready: bool = True) -> None:
    _write_json(
        project_root / "governance" / "health" / "runtime_throttle_control_latest.json",
        {
            "overall_status": "ready" if ready else "advisory",
            "host_saturation_score": 32.0 if ready else 58.0,
            "compute_pressure_level": "normal" if ready else "elevated",
            "memory_pressure_level": "normal",
            "mac_fluidity_contract": {
                "overall_status": "ready" if ready else "watch",
                "fluidity_score": 94.0 if ready else 82.0,
                "support_pause_recommended": False if ready else True,
            },
        },
    )
    _write_json(project_root / "governance" / "health" / "health_fast_latest.json", {"strict_all_clear": ready})


def _source_payload() -> dict:
    opsctl = "/repo/scripts/ops/opsctl.sh"
    return {
        "ok": False,
        "overall_status": "degraded",
        "unverified_sources": ["macro_crossstack", "ticker_news_context", "sec_edgar_context"],
        "stale_artifacts": ["ticker_news_context", "sec_edgar_context"],
        "degraded_artifacts": ["macro_crossstack", "ticker_news_context", "sec_edgar_context"],
        "recommended_refresh_commands": [
            [opsctl, "macro-crosscheck", "--json"],
            [opsctl, "ticker-news-sync", "--max-runtime-seconds", "240", "--json"],
            [opsctl, "sec-edgar-sync", "--json"],
            [opsctl, "source-verification", "--json"],
        ],
    }


def test_source_verification_autorefresh_selects_bounded_batch_when_runtime_ready(tmp_path: Path, monkeypatch) -> None:
    _seed_runtime(tmp_path, ready=True)
    monkeypatch.setattr(src.report_src, "build_source_verification_payload", lambda _root: _source_payload())

    payload = src.build_payload(tmp_path, apply=False, max_commands=8, timeout_seconds=180, max_heavy_commands=1)

    assert payload["overall_status"] == "needs_refresh"
    assert payload["runtime_refresh_contract"]["heavy_refresh_allowed"] is True
    assert [cmd[1] for cmd in payload["selected_commands"]] == ["macro-crosscheck", "ticker-news-sync"]
    selected = payload["selected_commands"][1]
    assert selected[1] == "ticker-news-sync"
    assert selected[selected.index("--max-symbols") + 1] == "300"
    assert selected[selected.index("--limit-per-symbol") + 1] == "6"
    assert selected[selected.index("--timeout-seconds") + 1] == "2.5"
    assert selected[selected.index("--max-runtime-seconds") + 1] == "150"
    assert selected.count("--max-runtime-seconds") == 1
    assert payload["skipped_commands"][0]["reason"] == "bounded_batch_cap"


def test_source_verification_autorefresh_prioritizes_decision_critical_sources(tmp_path: Path, monkeypatch) -> None:
    _seed_runtime(tmp_path, ready=True)
    opsctl = "/repo/scripts/ops/opsctl.sh"
    source_payload = {
        "ok": False,
        "overall_status": "degraded",
        "unverified_sources": ["ticker_news_context", "market_micro_context"],
        "stale_artifacts": ["ticker_news_context", "market_micro_context"],
        "degraded_artifacts": ["ticker_news_context", "market_micro_context"],
        "sources": [
            {"source_id": "ticker_news_context", "criticality": "decision_context", "fresh": False, "ok": False},
            {"source_id": "market_micro_context", "criticality": "decision_critical", "fresh": False, "ok": False},
        ],
        "recommended_refresh_commands": [
            [opsctl, "ticker-news-sync", "--json"],
            [opsctl, "market-micro-sync", "--json"],
            [opsctl, "source-verification", "--json"],
        ],
    }
    monkeypatch.setattr(src.report_src, "build_source_verification_payload", lambda _root: source_payload)

    payload = src.build_payload(tmp_path, apply=False, max_commands=1, timeout_seconds=180, max_heavy_commands=1)

    assert [cmd[1] for cmd in payload["selected_commands"]] == ["market-micro-sync"]
    assert payload["selection_contract"]["priority_order"] == [
        "decision_critical",
        "starvation_override_within_criticality",
        "decision_context",
        "optional_enrichment",
    ]


def test_source_verification_autorefresh_defers_heavy_sources_under_runtime_pressure(tmp_path: Path, monkeypatch) -> None:
    _seed_runtime(tmp_path, ready=False)
    heavy_only = _source_payload()
    heavy_only["recommended_refresh_commands"] = [
        heavy_only["recommended_refresh_commands"][1],
        heavy_only["recommended_refresh_commands"][2],
        heavy_only["recommended_refresh_commands"][3],
    ]
    monkeypatch.setattr(src.report_src, "build_source_verification_payload", lambda _root: heavy_only)

    payload = src.build_payload(tmp_path, apply=False, max_commands=8, timeout_seconds=180, max_heavy_commands=1)

    assert payload["overall_status"] == "deferred_by_runtime_governor"
    assert payload["selected_commands"] == []
    assert payload["runtime_refresh_contract"]["heavy_refresh_allowed"] is False
    assert {row["reason"] for row in payload["skipped_commands"]} == {
        "runtime_or_mac_fluidity_not_ready_for_heavy_refresh"
    }


def test_source_verification_autorefresh_allows_one_guarded_heavy_refresh_when_strict_clear(tmp_path: Path, monkeypatch) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "host_saturation_score": 43.2,
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "pressure_pause_bypassed": True,
            },
            "mac_fluidity_contract": {
                "overall_status": "needs_work",
                "fluidity_score": 72.5,
                "support_pause_recommended": False,
            },
        },
    )
    _write_json(tmp_path / "governance" / "health" / "health_fast_latest.json", {"strict_all_clear": True})
    heavy_only = _source_payload()
    heavy_only["recommended_refresh_commands"] = [
        heavy_only["recommended_refresh_commands"][1],
        heavy_only["recommended_refresh_commands"][2],
        heavy_only["recommended_refresh_commands"][3],
    ]
    monkeypatch.setattr(src.report_src, "build_source_verification_payload", lambda _root: heavy_only)

    payload = src.build_payload(tmp_path, apply=False, max_commands=8, timeout_seconds=180, max_heavy_commands=1)

    assert payload["overall_status"] == "needs_refresh"
    assert payload["runtime_refresh_contract"]["heavy_refresh_allowed"] is True
    assert payload["runtime_refresh_contract"]["heavy_refresh_mode"] == "guarded_single_heavy"
    assert [cmd[1] for cmd in payload["selected_commands"]] == ["ticker-news-sync"]
    selected = payload["selected_commands"][0]
    assert selected[selected.index("--max-symbols") + 1] == "300"
    assert selected[selected.index("--limit-per-symbol") + 1] == "6"
    assert selected[selected.index("--timeout-seconds") + 1] == "2.5"
    assert selected[selected.index("--max-runtime-seconds") + 1] == "150"
    assert selected.count("--max-runtime-seconds") == 1
    assert payload["skipped_commands"][0]["reason"] == "bounded_batch_cap"


def test_source_verification_autorefresh_refreshes_macro_dependencies_before_crosscheck(tmp_path: Path, monkeypatch) -> None:
    _seed_runtime(tmp_path, ready=True)
    opsctl = "/repo/scripts/ops/opsctl.sh"
    source_payload = {
        "ok": False,
        "overall_status": "degraded",
        "unverified_sources": ["macro_crossstack", "public_macro_feeds", "market_micro_context"],
        "stale_artifacts": ["public_macro_feeds", "market_micro_context"],
        "degraded_artifacts": ["macro_crossstack", "public_macro_feeds", "market_micro_context"],
        "recommended_refresh_commands": [
            [opsctl, "macro-crosscheck", "--json"],
            [opsctl, "macro-context-sync", "--json"],
            [opsctl, "market-micro-sync", "--json"],
            [opsctl, "source-verification", "--json"],
        ],
    }
    monkeypatch.setattr(src.report_src, "build_source_verification_payload", lambda _root: source_payload)

    payload = src.build_payload(tmp_path, apply=False, max_commands=1, timeout_seconds=180, max_heavy_commands=1)

    assert [cmd[1] for cmd in payload["selected_commands"]] == ["macro-context-sync"]
    skipped_by_reason = {row["reason"]: row for row in payload["skipped_commands"]}
    assert skipped_by_reason["dependent_stale_sources_waiting"]["command"][1] == "macro-crosscheck"
    assert skipped_by_reason["dependent_stale_sources_waiting"]["stale_dependencies"] == [
        "market_micro_context",
        "public_macro_feeds",
    ]


def test_source_verification_autorefresh_does_not_pin_on_fresh_confidence_debt(tmp_path: Path, monkeypatch) -> None:
    _seed_runtime(tmp_path, ready=True)
    opsctl = "/repo/scripts/ops/opsctl.sh"
    source_payload = {
        "ok": False,
        "overall_status": "degraded",
        "unverified_sources": ["fx_market_context", "public_macro_feeds"],
        "stale_artifacts": ["public_macro_feeds"],
        "degraded_artifacts": ["fx_market_context", "public_macro_feeds"],
        "sources": [
            {"source_id": "fx_market_context", "fresh": True, "ok": True},
            {"source_id": "public_macro_feeds", "fresh": False, "ok": False},
        ],
        "recommended_refresh_commands": [
            [opsctl, "fx-market-sync", "--json"],
            [opsctl, "macro-context-sync", "--json"],
            [opsctl, "source-verification", "--json"],
        ],
    }
    monkeypatch.setattr(src.report_src, "build_source_verification_payload", lambda _root: source_payload)

    payload = src.build_payload(tmp_path, apply=False, max_commands=1, timeout_seconds=180, max_heavy_commands=1)

    assert [cmd[1] for cmd in payload["selected_commands"]] == ["macro-context-sync"]
    skipped_by_reason = {row["reason"]: row for row in payload["skipped_commands"]}
    assert skipped_by_reason["fresh_ok_source_confidence_debt_waiting"]["source_id"] == "fx_market_context"


def test_source_verification_autorefresh_reconciles_downstream_reports_after_apply(tmp_path: Path, monkeypatch) -> None:
    _seed_runtime(tmp_path, ready=True)
    opsctl = "/repo/scripts/ops/opsctl.sh"
    before = {
        "ok": False,
        "overall_status": "degraded",
        "unverified_sources": ["sec_edgar_context"],
        "stale_artifacts": ["sec_edgar_context"],
        "degraded_artifacts": ["sec_edgar_context"],
        "recommended_refresh_commands": [
            [opsctl, "sec-edgar-sync", "--json"],
            [opsctl, "source-verification", "--json"],
        ],
    }
    after = {
        "ok": True,
        "overall_status": "ready",
        "unverified_sources": [],
        "stale_artifacts": [],
        "degraded_artifacts": [],
        "recommended_refresh_commands": [[opsctl, "source-verification", "--json"]],
    }
    payloads = iter([before, after])
    calls: list[list[str]] = []

    def _fake_run(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict:
        calls.append(list(command))
        return {
            "command": list(command),
            "rc": 0,
            "ok": True,
            "stdout_tail": "",
            "stderr_tail": "",
            "timed_out": False,
        }

    monkeypatch.setattr(src.report_src, "build_source_verification_payload", lambda _root: next(payloads))
    monkeypatch.setattr(src, "_write_latest_source_report", lambda _root, _payload: None)
    monkeypatch.setattr(src, "_run_command", _fake_run)

    payload = src.build_payload(tmp_path, apply=True, max_commands=1, timeout_seconds=180, max_heavy_commands=1)

    assert payload["overall_status"] == "applied"
    assert [cmd[1] for cmd in calls] == ["sec-edgar-sync", "collector-contracts", "health-gates"]
    assert [cmd[1] for cmd in payload["downstream_recheck_commands"]] == ["collector-contracts", "health-gates"]
    assert [row["command"][1] for row in payload["downstream_recheck_results"]] == ["collector-contracts", "health-gates"]


def test_source_verification_autorefresh_tails_are_bounded() -> None:
    assert len(src._tail_text("x" * 5000, char_limit=123)) == 123


def test_source_verification_autorefresh_persists_failure_backoff(tmp_path: Path, monkeypatch) -> None:
    _seed_runtime(tmp_path, ready=True)
    opsctl = "/repo/scripts/ops/opsctl.sh"
    source_payload = {
        "ok": False,
        "overall_status": "degraded",
        "unverified_sources": ["sec_edgar_context"],
        "stale_artifacts": ["sec_edgar_context"],
        "degraded_artifacts": ["sec_edgar_context"],
        "recommended_refresh_commands": [
            [opsctl, "sec-edgar-sync", "--json"],
            [opsctl, "source-verification", "--json"],
        ],
    }

    def _failed(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict:
        return {
            "command": list(command),
            "rc": 1,
            "ok": False,
            "stdout_tail": "",
            "stderr_tail": "failed",
            "timed_out": False,
        }

    monkeypatch.setattr(src.report_src, "build_source_verification_payload", lambda _root: source_payload)
    monkeypatch.setattr(src, "_write_latest_source_report", lambda _root, _payload: None)
    monkeypatch.setattr(src, "_run_command", _failed)
    state_path = tmp_path / "governance" / "runtime" / "retry.json"

    first = src.build_payload(tmp_path, apply=True, max_commands=1, state_path=state_path)
    second = src.build_payload(tmp_path, apply=False, max_commands=1, state_path=state_path)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert first["results"][0]["source_id"] == "sec_edgar_context"
    assert state["sources"]["sec_edgar_context"]["consecutive_failures"] == 1
    assert second["selected_commands"] == []
    assert second["skipped_commands"][0]["reason"] == "retry_backoff_active"
    assert second["overall_status"] == "deferred_by_retry_backoff"
    assert second["ok"] is True


def test_source_retry_quarantine_is_bounded_and_starvation_safe() -> None:
    state = {"schema_version": 1, "sources": {}}
    started = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    failure = {"ok": False, "rc": 1, "timed_out": False}
    for index in range(src.RETRY_QUARANTINE_FAILURES):
        src._record_retry_result(
            state,
            source_id="market_quote_profiles",
            result=failure,
            now=started + timedelta(minutes=index),
        )

    quarantined = state["sources"]["market_quote_profiles"]
    decision = src._retry_decision(
        "market_quote_profiles",
        state,
        now=started + timedelta(hours=7),
    )

    assert quarantined["quarantined"] is True
    assert decision["due"] is True
    assert decision["starvation_override"] is True


def test_source_refresh_backs_off_when_process_succeeds_but_evidence_remains_unverified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _seed_runtime(tmp_path, ready=True)
    opsctl = "/repo/scripts/ops/opsctl.sh"
    before = {
        "ok": False,
        "overall_status": "degraded",
        "unverified_sources": ["public_policy_context"],
        "stale_artifacts": [],
        "degraded_artifacts": ["public_policy_context"],
        "sources": [
            {
                "source_id": "public_policy_context",
                "fresh": True,
                "ok": False,
                "verification_status": "single_source_unverified",
            }
        ],
        "recommended_refresh_commands": [
            [opsctl, "public-policy-sync", "--json"],
            [opsctl, "source-verification", "--json"],
        ],
    }
    after = dict(before)
    payloads = iter([before, after])
    monkeypatch.setattr(src.report_src, "build_source_verification_payload", lambda _root: next(payloads))
    monkeypatch.setattr(src, "_write_latest_source_report", lambda _root, _payload: None)
    monkeypatch.setattr(
        src,
        "_run_command",
        lambda command, **_kwargs: {
            "command": list(command),
            "rc": 0,
            "ok": True,
            "stdout_tail": "",
            "stderr_tail": "",
            "timed_out": False,
        },
    )
    state_path = tmp_path / "governance" / "runtime" / "retry.json"

    payload = src.build_payload(tmp_path, apply=True, max_commands=1, state_path=state_path)

    result = payload["results"][0]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "applied_with_failures"
    assert result["collector_ok"] is True
    assert result["source_evidence_ok"] is False
    assert result["ok"] is False
    assert result["semantic_failure"]
    assert state["sources"]["public_policy_context"]["consecutive_failures"] == 1
    assert state["sources"]["public_policy_context"]["last_failure_kind"] == "semantic_evidence_incomplete"
    assert state["sources"]["public_policy_context"]["retry_delay_seconds"] >= src.SEMANTIC_RETRY_MIN_SECONDS
