import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/crash_report_digest.py')
spec = importlib.util.spec_from_file_location('crash_report_digest', MODULE_PATH)
crash_report_digest = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(crash_report_digest)


def test_halt_windows_infer_missing_reason_root_cause():
    rows = [
        {
            'timestamp_utc': '2026-03-10T22:52:27.661882Z',
            'action': 'halt_auto_clear_skipped',
            'decision_reason': 'cooldown_not_elapsed:3.5s<60s',
            'halt_active': True,
            'halt_reason': '',
        },
        {
            'timestamp_utc': '2026-03-10T22:52:47.891461Z',
            'action': 'halt_auto_clear_skipped',
            'decision_reason': 'cooldown_not_elapsed:23.7s<60s',
            'halt_active': True,
            'halt_reason': '',
        },
        {
            'timestamp_utc': '2026-03-10T22:53:28.242848Z',
            'action': 'halt_auto_clear_skipped',
            'decision_reason': 'reason_not_allowed:unknown',
            'halt_active': True,
            'halt_reason': '',
        },
        {
            'timestamp_utc': '2026-03-10T22:53:48.335215Z',
            'action': 'halt_auto_clear_skipped',
            'decision_reason': 'reason_not_allowed:unknown',
            'halt_active': True,
            'halt_reason': '',
        },
    ]

    counts, windows = crash_report_digest._halt_recovery_windows(rows, auto_halt_rows=[])

    assert counts['halt_auto_clear_skipped|cooldown_not_elapsed'] == 2
    assert counts['halt_auto_clear_skipped|reason_not_allowed:unknown'] == 2
    assert len(windows) == 2
    assert windows[0]['root_cause'] == 'missing_halt_reason'
    assert 'no usable reason' in windows[0]['root_detail']
    assert windows[0]['count'] == 2


def test_halt_windows_infer_softguard_root_cause_and_normalize_cooldown():
    rows = [
        {
            'timestamp_utc': '2026-03-10T22:01:42.958141Z',
            'action': 'halt_auto_clear_skipped',
            'decision_reason': 'cooldown_not_elapsed:2.7s<60s',
            'halt_active': True,
            'halt_reason': 'softguard_api_circuit_opened',
        },
        {
            'timestamp_utc': '2026-03-10T22:02:03.151728Z',
            'action': 'halt_auto_clear_skipped',
            'decision_reason': 'cooldown_not_elapsed:22.9s<60s',
            'halt_active': True,
            'halt_reason': 'softguard_api_circuit_opened',
        },
        {
            'timestamp_utc': '2026-03-10T22:02:23.352227Z',
            'action': 'halt_auto_clear_skipped',
            'decision_reason': 'cooldown_not_elapsed:43.1s<60s',
            'halt_active': True,
            'halt_reason': 'softguard_api_circuit_opened',
        },
        {
            'timestamp_utc': '2026-03-10T22:02:43.462361Z',
            'action': 'halt_auto_cleared',
            'decision_reason': 'eligible',
            'halt_active': False,
            'halt_reason': 'softguard_api_circuit_opened',
        },
    ]

    counts, windows = crash_report_digest._halt_recovery_windows(rows, auto_halt_rows=[])
    root_counts, examples = crash_report_digest._summarize_halt_root_causes(windows)

    assert counts['halt_auto_clear_skipped|cooldown_not_elapsed'] == 3
    assert counts['halt_auto_cleared|eligible'] == 1
    assert windows[1]['root_cause'] == 'softguard_api_circuit_opened'
    assert 'still inside cooldown' in windows[1]['root_detail']
    assert root_counts['softguard_api_circuit_opened'] == 2
    assert 'softguard_api_circuit_opened' in examples['softguard_api_circuit_opened']


def test_collect_critical_alert_rows_dedupes_latest_and_classifies_limit_hits(tmp_path, monkeypatch):
    alerts_dir = tmp_path / 'alerts'
    alerts_dir.mkdir()

    payload = {
        'timestamp_utc': '2026-03-12T12:28:45.732650+00:00',
        'severity': 'warn',
        'event': 'options_margin_guard',
        'message': 'options_margin_headroom_insufficient',
        'broker': 'coinbase',
        'profile': 'default',
        'domain': 'crypto',
        'details': {
            'symbol': 'LTC-USD',
            'required_margin_proxy': 273.0,
            'available_margin_proxy': 0.0,
        },
    }
    (alerts_dir / 'critical_events_20260312.jsonl').write_text(json.dumps(payload) + '\n', encoding='utf-8')
    (alerts_dir / 'critical_latest_default_crypto_coinbase.json').write_text(json.dumps(payload), encoding='utf-8')

    monkeypatch.setattr(crash_report_digest, 'ALERTS_DIR', alerts_dir)
    since_utc = crash_report_digest._parse_ts('2026-03-12T00:00:00Z')
    assert since_utc is not None

    rows = crash_report_digest._collect_critical_alert_rows(since_utc)
    counts, incidents = crash_report_digest._critical_alert_summary(rows)

    assert len(rows) == 1
    assert counts['Default / Coinbase|Options Margin Guard'] == 1
    assert incidents[0]['target'] == 'Default / Coinbase'
    assert 'symbol=LTC-USD' in incidents[0]['detail']
    assert crash_report_digest._incident_class(incidents[0]) == 'bot_limit_guardrail'


def test_partition_incidents_by_class_separates_limits_crashes_and_other_alerts():
    incidents = [
        {
            'timestamp_utc': '2026-03-12T12:30:00+00:00',
            'source': 'critical_alert',
            'target': 'Default / Coinbase',
            'event': 'Options Margin Guard',
            'event_key': 'options_margin_guard',
            'detail': 'options_margin_headroom_insufficient',
        },
        {
            'timestamp_utc': '2026-03-12T12:29:00+00:00',
            'source': 'watchdog',
            'target': 'all_sleeves',
            'event': 'restart',
            'detail': 'restart_attempted',
        },
        {
            'timestamp_utc': '2026-03-12T12:28:00+00:00',
            'source': 'critical_alert',
            'target': 'Dividend / Schwab',
            'event': 'Broker Truth Reconcile',
            'event_key': 'broker_truth_reconcile',
            'detail': 'error source=schwab_accounts_snapshot',
        },
    ]

    class_counts, recent_limit_incidents, recent_crash_incidents, recent_other_incidents = crash_report_digest._partition_incidents_by_class(
        incidents,
        recent_limit=10,
    )

    assert class_counts['bot_limit_guardrail'] == 1
    assert class_counts['crash_restart'] == 1
    assert class_counts['operational_alert'] == 1
    assert recent_limit_incidents[0]['source'] == 'critical_alert'
    assert recent_limit_incidents[0]['incident_class'] == 'bot_limit_guardrail'
    assert recent_crash_incidents[0]['source'] == 'watchdog'
    assert recent_crash_incidents[0]['incident_class'] == 'crash_restart'
    assert recent_other_incidents[0]['event_key'] == 'broker_truth_reconcile'
    assert recent_other_incidents[0]['incident_class'] == 'operational_alert'


def test_lane_pause_alerts_are_classified_as_bot_limit_guardrails():
    incident = {
        'timestamp_utc': '2026-03-12T12:31:00+00:00',
        'source': 'critical_alert',
        'target': 'Intraday Aggressive / Schwab',
        'event': 'Lane Consecutive Loss Pause',
        'event_key': 'lane_consecutive_loss_pause',
        'detail': 'loss_streak=3 cooldown=900s',
    }

    assert crash_report_digest._incident_class(incident) == 'bot_limit_guardrail'
