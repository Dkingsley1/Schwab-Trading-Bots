import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/system_explainer_docs.py')
spec = importlib.util.spec_from_file_location('system_explainer_docs', MODULE_PATH)
system_explainer_docs = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(system_explainer_docs)


def test_build_docs_writes_expected_outputs(tmp_path):
    health_dir = tmp_path / 'governance' / 'health'
    feature_store_dir = tmp_path / 'governance' / 'feature_store'
    reports_dir = tmp_path / 'exports' / 'reports' / 'system_explainers'
    health_dir.mkdir(parents=True)
    feature_store_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    (health_dir / 'collector_contracts_latest.json').write_text(json.dumps({
        'collector_count': 11,
        'required_failure_count': 0,
        'soft_failure_count': 0,
        'average_quality_score': 0.95,
    }), encoding='utf-8')
    (health_dir / 'source_verification_latest.json').write_text(json.dumps({
        'overall': {'all_verified': True, 'counts': {'cross_verified': 3}},
    }), encoding='utf-8')
    (health_dir / 'ingestion_storage_control_latest.json').write_text(json.dumps({
        'overall_status': 'ready',
        'recommended_operating_mode': 'live_full',
        'pressure_index': 0.04,
        'backpressure': {'core_pending_lines': 12},
        'steady_state': {
            'quality_score': 98.1,
            'targets': {
                'pressure_index': 0.25,
                'core_pending_lines': 5000,
                'estimated_total_drain_minutes': 15.0,
            },
            'target_status': {'steady_state_ready': True},
        },
    }), encoding='utf-8')
    (health_dir / 'health_gates_latest.json').write_text(json.dumps({
        'recommended_operating_mode': 'live_full',
        'data_quality_score': 91.2,
        'hard_gate_triggered': False,
    }), encoding='utf-8')
    (health_dir / 'sql_link_service_latest.json').write_text(json.dumps({
        'mode': 'sharded_merge',
        'link_mode': 'sqlite',
        'primary_db_realpath': '/Volumes/BOT_LOGS/schwab_trading_bot/data/jsonl_link.sqlite3',
    }), encoding='utf-8')
    (health_dir / 'paper_performance_latest.json').write_text(json.dumps({
        'active_paper_profiles_today': [
            {'profile': 'intraday_aggressive'},
            {'profile': 'fx'},
            {'profile': 'schwab_futures'},
            {'profile': 'dividend_capture'},
        ],
    }), encoding='utf-8')
    (feature_store_dir / 'latest.json').write_text(json.dumps({
        'ok': True,
        'strict_ok': False,
        'dataset_contract': {'row_count': 1234},
        'point_in_time_contract': {'event_category_count': 7, 'event_count': 42, 'event_store_fresh': True},
    }), encoding='utf-8')

    outputs = system_explainer_docs.build_docs(tmp_path)

    assert 'framework_map_v2' in outputs
    assert 'runtime_hierarchy' in outputs
    assert 'training_and_promotion' in outputs

    framework_path = Path(outputs['framework_map_v2'])
    runtime_path = Path(outputs['runtime_hierarchy'])
    manifest_path = tmp_path / 'governance' / 'health' / 'system_explainer_docs_latest.json'

    assert framework_path.exists()
    assert runtime_path.exists()
    assert manifest_path.exists()

    framework_text = framework_path.read_text(encoding='utf-8')
    runtime_text = runtime_path.read_text(encoding='utf-8')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    assert 'Framework Map v2' in framework_text
    assert 'Operational Architecture Report' in framework_text
    assert 'Executive Summary' in framework_text
    assert 'How To Read This Report' in framework_text
    assert 'Architecture Recommendations' in framework_text
    assert 'Runtime Hierarchy' in runtime_text
    assert 'dividend_capture' in framework_text
    assert 'Backpressure score' in framework_text
    assert 'Backpressure scorecard' in Path(outputs['data_intake_and_shards']).read_text(encoding='utf-8')
    assert 'Active paper lane roster today' in runtime_text
    assert manifest['schema_version'] == 1
    assert 'files' in manifest
