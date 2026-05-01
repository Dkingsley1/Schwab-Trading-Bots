import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/training_report.py')
spec = importlib.util.spec_from_file_location('training_report', MODULE_PATH)
training_report = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(training_report)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding='utf-8')


def test_build_context_uses_lineage_trade_log_and_computes_score_delta(tmp_path):
    scorecard_path = tmp_path / 'retrain_scorecard_latest.json'
    training_success_path = tmp_path / 'training_success_latest.json'
    promotion_quality_path = tmp_path / 'promotion_quality_gate_latest.json'
    promotion_gate_path = tmp_path / 'promotion_gate_latest.json'
    graduation_path = tmp_path / 'new_bot_graduation_latest.json'
    daily_verify_path = tmp_path / 'daily_auto_verify_latest.json'
    divergence_path = tmp_path / 'data_source_divergence_latest.json'
    lane_scorecard_path = tmp_path / 'unified_lane_scorecard_latest.json'
    trade_log_path = tmp_path / 'trade_behavior_policy_20260312_155116.json'

    _write_json(scorecard_path, {
        'timestamp_utc': '2026-03-12T15:51:23.740633+00:00',
        'started_utc': '2026-03-12T15:39:35+00:00',
        'ended_utc': '2026-03-12T15:48:08+00:00',
        'target_count': 9,
        'status_counts': {'trained': 9},
        'accuracy_delta': {'improved': 0, 'degraded': 0, 'unchanged': 96},
        'master_update_status': 'precheck_failed',
        'lineage': {'trade_behavior_log_latest': str(trade_log_path)},
    })
    _write_json(training_success_path, {
        'timestamp_utc': '2026-03-12T15:48:08.116597+00:00',
        'trained_count': 9,
        'failure_count': 0,
        'confirmed_training_success': False,
        'reason': 'master_update_not_updated:precheck_failed',
        'master_update_status': 'precheck_failed',
        'data_quality_ok': False,
    })
    _write_json(promotion_quality_path, {
        'ok': False,
        'failed_checks': ['promotion_gate_blocked', 'daily_verify_not_ok'],
    })
    _write_json(promotion_gate_path, {
        'promote_ok': False,
        'coverage_ok': False,
        'considered_bots': 4,
        'failed_bots': 0,
        'fail_share': 0.0,
        'mean_trading_quality_score': 0.969962,
        'thresholds': {'min_considered_bots': 12},
    })
    _write_json(graduation_path, {'ok': False, 'immature_active_count': 3, 'maturity': {'mature_pass_rate': 0.363636}})
    _write_json(daily_verify_path, {'ok': False, 'failed_checks': ['daily_verify_not_ok']})
    _write_json(divergence_path, {
        'ok': False,
        'window_hours': 24,
        'worst_relative_spread': 137.432683,
        'max_relative_spread': 0.03,
        'offenders': [{'symbol': 'TLT', 'minute': '2026-03-12T13:45:00Z', 'rel_spread': 137.432683, 'n': 12}],
    })
    _write_json(lane_scorecard_path, {
        'ok': True,
        'lookback_hours': 24,
        'rows_used': 1024,
        'lanes': {
            'equities_day': {
                'rows': 128,
                'hold_rows': 33,
                'avg_master_score': 0.9123,
                'execution_guard_block_rate': 0.12,
                'portfolio_risk_block_rate': 0.05,
            }
        },
    })
    _write_json(trade_log_path, {
        'candidate_score': 1.2227,
        'previous_score': 1.2631,
        'promoted': False,
        'deployed_from_previous': True,
        'deployed_previous_model': 'models/trade_behavior_policy_20260312_012053.npz',
        'model_path': 'models/trade_behavior_policy_20260312_155116.npz',
        'promotion_gate': {'reasons': ['accuracy_below_threshold', 'macro_f1_below_threshold']},
        'test_metrics': {'accuracy': 0.2894, 'macro_f1': 0.2844, 'balanced_accuracy': 0.3481},
    })

    context = training_report._build_context(
        scorecard_path=scorecard_path,
        training_success_path=training_success_path,
        promotion_quality_path=promotion_quality_path,
        promotion_gate_path=promotion_gate_path,
        graduation_path=graduation_path,
        daily_verify_path=daily_verify_path,
        data_divergence_path=divergence_path,
        lane_scorecard_path=lane_scorecard_path,
    )

    assert context['trade_behavior']['log_path'] == str(trade_log_path)
    assert context['trade_behavior']['score_delta'] == pytest.approx(-0.0404, abs=1e-9)
    assert context['promotion_gate']['considered_bots'] == 4
    assert context['lane_scorecard']['lane_rows'][0]['lane'] == 'equities_day'
    assert context['summary']['master_update_status'] == 'precheck_failed'


def test_assessment_lines_call_out_failed_run_and_rollback():
    context = {
        'summary': {
            'target_count': 9,
            'trained_count': 9,
            'confirmed_training_success': False,
            'training_reason': 'master_update_not_updated:precheck_failed',
            'master_update_status': 'precheck_failed',
        },
        'promotion_quality': {
            'failed_checks': ['promotion_gate_blocked', 'daily_verify_not_ok'],
        },
        'promotion_gate': {
            'coverage_ok': False,
            'considered_bots': 4,
            'thresholds': {'min_considered_bots': 12},
        },
        'trade_behavior': {
            'candidate_score': 1.2227,
            'previous_score': 1.2631,
            'score_delta': -0.0404,
            'promoted': False,
            'deployed_from_previous': True,
        },
        'data_divergence': {
            'ok': False,
            'worst_relative_spread': 137.432683,
            'max_relative_spread': 0.03,
        },
    }

    lines = training_report._assessment_lines(context)

    assert any('not confirmed successful' in line for line in lines)
    assert any('Promotion quality gate blockers' in line for line in lines)
    assert any('Walk-forward coverage is still short' in line for line in lines)
    assert any('Trade behavior candidate regressed' in line for line in lines)
    assert any('previous deployed trade behavior model remained active after rollback' in line for line in lines)
    assert any('Data divergence is above the allowed threshold' in line for line in lines)


def test_blocking_reasons_and_overall_status_capture_regression_and_divergence():
    context = {
        'summary': {
            'target_count': 1,
            'trained_count': 0,
            'failure_count': 1,
            'confirmed_training_success': False,
            'training_reason': 'precheck_failed',
            'promotion_quality_ok': False,
            'daily_verify_ok': False,
        },
        'promotion_quality': {
            'failed_checks': ['promotion_gate_blocked', 'daily_verify_not_ok'],
        },
        'promotion_gate': {
            'promote_ok': False,
        },
        'data_divergence': {
            'ok': False,
        },
        'trade_behavior': {
            'score_delta': -0.02,
        },
    }

    reasons = training_report._blocking_reasons(context)
    status = training_report._overall_status(context, reasons)

    assert reasons == [
        'training_not_confirmed',
        'promotion_gate_blocked',
        'daily_verify_not_ok',
        'promotion_not_ready',
        'data_divergence_not_ok',
        'trade_behavior_regressed',
    ]
    assert status == 'blocked'


def test_build_context_includes_training_quality_control(tmp_path):
    scorecard_path = tmp_path / 'retrain_scorecard_latest.json'
    training_success_path = tmp_path / 'training_success_latest.json'
    promotion_quality_path = tmp_path / 'promotion_quality_gate_latest.json'
    promotion_gate_path = tmp_path / 'promotion_gate_latest.json'
    graduation_path = tmp_path / 'new_bot_graduation_latest.json'
    daily_verify_path = tmp_path / 'daily_auto_verify_latest.json'
    divergence_path = tmp_path / 'data_source_divergence_latest.json'
    lane_scorecard_path = tmp_path / 'unified_lane_scorecard_latest.json'
    training_quality_control_path = tmp_path / 'training_quality_control_latest.json'

    _write_json(scorecard_path, {'timestamp_utc': '2026-03-12T15:51:23.740633+00:00'})
    _write_json(training_success_path, {'timestamp_utc': '2026-03-12T15:48:08.116597+00:00', 'trained_count': 1, 'confirmed_training_success': False})
    _write_json(promotion_quality_path, {'ok': False, 'failed_checks': ['promotion_gate_blocked']})
    _write_json(promotion_gate_path, {'promote_ok': False, 'coverage_ok': False, 'considered_bots': 1})
    _write_json(graduation_path, {'ok': False, 'immature_active_count': 0})
    _write_json(daily_verify_path, {'ok': False, 'failed_checks': ['daily_verify_not_ok']})
    _write_json(divergence_path, {'ok': True, 'window_hours': 24})
    _write_json(lane_scorecard_path, {'ok': True, 'lookback_hours': 24, 'rows_used': 10, 'lanes': {}})
    _write_json(
        training_quality_control_path,
        {
            'ok': False,
            'overall_status': 'blocked',
            'training_quality_index': 67.5,
            'training_quality_base_score': 61.5,
            'training_quality_bonus_score': 6.0,
            'training_quality_score': 61.5,
            'top_priorities': ['runtime_input_coverage', 'stale_active_diagnostics'],
            'implemented_improvement_count': 17,
            'immutable_lineage': {'lineage_status': 'blocked'},
            'failure_taxonomy': {'failure_buckets': ['runtime_input_gap', 'storage_backpressure']},
        },
    )

    context = training_report._build_context(
        scorecard_path=scorecard_path,
        training_success_path=training_success_path,
        promotion_quality_path=promotion_quality_path,
        promotion_gate_path=promotion_gate_path,
        graduation_path=graduation_path,
        daily_verify_path=daily_verify_path,
        data_divergence_path=divergence_path,
        lane_scorecard_path=lane_scorecard_path,
        training_quality_control_path=training_quality_control_path,
    )

    assert context['training_quality_control']['overall_status'] == 'blocked'
    assert context['training_quality_control']['training_quality_index'] == 67.5
    assert context['training_quality_control']['implemented_improvement_count'] == 17
    assert context['immutable_lineage']['lineage_status'] == 'blocked'
    assert any('Training quality control is blocked' in line for line in context['assessment'])
    assert any('Immutable lineage is still blocked' in line for line in context['assessment'])


def test_passive_cycle_is_not_treated_as_training_not_confirmed():
    context = {
        'summary': {
            'target_count': 0,
            'trained_count': 0,
            'failure_count': 0,
            'confirmed_training_success': False,
            'training_reason': '',
            'master_update_status': '',
            'promotion_quality_ok': False,
            'daily_verify_ok': False,
        },
        'promotion_quality': {'failed_checks': ['promotion_gate_blocked']},
        'promotion_gate': {'promote_ok': False, 'coverage_ok': False, 'considered_bots': 0, 'thresholds': {'min_considered_bots': 4}},
        'trade_behavior': {'score_delta': None, 'promoted': False, 'deployed_from_previous': False},
        'data_divergence': {'ok': True},
    }

    lines = training_report._assessment_lines(context)
    reasons = training_report._blocking_reasons(context)

    assert any('No active retrain batch is recorded right now' in line for line in lines)
    assert 'training_not_confirmed' not in reasons
