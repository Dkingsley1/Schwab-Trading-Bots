import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/live_runtime_separation_control.py')
spec = importlib.util.spec_from_file_location('live_runtime_separation_control', MODULE_PATH)
live_runtime_separation_control = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(live_runtime_separation_control)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding='utf-8')


def test_guarded_paper_soak_does_not_require_live_lane(tmp_path):
    health = tmp_path / 'governance' / 'health'
    walk = tmp_path / 'governance' / 'walk_forward'

    _write_json(health / 'live_readiness_smoke_latest.json', {'ok': False, 'broker_ready': True, 'session_ready': True, 'live_lane_running': False})
    _write_json(health / 'training_runtime_control_latest.json', {'overall_status': 'ready', 'snapshot_ready': True, 'precompute_targets': []})
    _write_json(health / 'storage_tier_policy_latest.json', {'overall_status': 'ready', 'pressure': {'hot_path_over_budget_bytes': 0}})
    _write_json(
        health / 'ingestion_storage_control_latest.json',
        {
            'overall_status': 'ready',
            'severity': 'stable',
            'pressure_index': 0.01,
            'backpressure_quality_score': 100.0,
            'recovery_quality_score': 100.0,
            'external_route_verification': {'verification_state': 'ready'},
            'steady_state': {'target_status': {'steady_state_ready': True, 'target_breaches': [], 'target_breach_count': 0}},
            'backpressure': {'raw_live': {'core_pending_lines': 0, 'total_pending_lines': 0, 'oldest_pending_age_seconds': 0.0}},
        },
    )
    _write_json(health / 'runtime_throttle_control_latest.json', {'overall_status': 'ready'})
    _write_json(health / 'resource_guard_latest.json', {'swap_used_gb': 0.0})
    _write_json(health / 'process_watchdog_latest.json', {'restart_storms': []})
    _write_json(health / 'unattended_soak_readiness_latest.json', {'overall_status': 'ready', 'safe_to_leave_unattended': True, 'overall_grade': 'A+'})
    _write_json(health / 'runtime_paper_regression_guard_latest.json', {'overall_status': 'ready', 'paper_armed': True, 'paper_blocked': False})
    _write_json(health / 'paper_400_ramp_latest.json', {'stage': 'armed', 'armed': True, 'blockers': []})
    _write_json(health / 'health_fast_latest.json', {'overall_status': 'ready', 'strict_all_clear': True})
    _write_json(walk / 'coverage_seed_latest.json', {'coverage_shortfall_bots': 0})

    payload = live_runtime_separation_control.build_payload(tmp_path)

    assert payload['overall_status'] == 'ready'
    assert payload['ok'] is True
    assert payload['live_plane']['ready'] is False
    assert payload['live_plane']['guarded_paper_soak_ready'] is True
    assert payload['live_plane']['guarded_paper_read_only_ready'] is True
    assert payload['release_contract']['live_lane_should_be_read_only'] is True
