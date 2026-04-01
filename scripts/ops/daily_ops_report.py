import argparse
import csv
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _proc_count(pattern: str, *, exclude: tuple[str, ...] = ()) -> int:
    p = subprocess.run(['ps', '-axo', 'command'], capture_output=True, text=True, check=False)
    out = p.stdout or ''
    return sum(
        1
        for line in out.splitlines()
        if pattern in line and not any(marker in line for marker in exclude)
    )


def _one_numbers() -> dict:
    summary = _read_json(PROJECT_ROOT / 'exports' / 'one_numbers' / 'one_numbers_summary.json')
    if summary:
        return summary

    latest_csv = PROJECT_ROOT / 'exports' / 'one_numbers' / 'latest.csv'
    out: dict = {}
    if latest_csv.exists():
        try:
            with latest_csv.open('r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    out[str(row.get('metric', ''))] = row.get('value')
        except Exception:
            pass
    return out


def _ingress_health(profile: str, *, domain: str = 'equities', broker: str = 'schwab') -> dict:
    return _read_json(PROJECT_ROOT / 'governance' / 'health' / f'data_ingress_latest_{profile}_{domain}_{broker}.json')


def _paper_performance() -> dict:
    return _read_json(PROJECT_ROOT / 'governance' / 'health' / 'paper_performance_latest.json')


def _sleeve_latest(payload: dict, profile: str) -> dict:
    rows = payload.get('sleeve_latest')
    if not isinstance(rows, list):
        return {}
    profile_name = str(profile or '').strip().lower()
    for row in rows:
        if isinstance(row, dict) and str(row.get('profile', '')).strip().lower() == profile_name:
            return row
    return {}


def _to_float(v: object, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return d


def _post_webhook(url: str, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode('utf-8')
    req = urllib.request.Request(url=url, data=body, method='POST', headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=8) as resp:
        _ = resp.read()


def main() -> int:
    parser = argparse.ArgumentParser(description='Build daily operator report and optional webhook push.')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    one = _one_numbers()
    readiness = _read_json(PROJECT_ROOT / 'governance' / 'walk_forward' / 'promotion_readiness_latest.json')
    verify = _read_json(PROJECT_ROOT / 'governance' / 'health' / 'daily_auto_verify_latest.json')
    source_verification = _read_json(PROJECT_ROOT / 'governance' / 'health' / 'source_verification_latest.json')
    paper_performance = _paper_performance()
    dividend_ingress = _ingress_health('dividend')
    dividend_latest = _sleeve_latest(paper_performance, 'dividend')

    disk = shutil.disk_usage(PROJECT_ROOT)
    free_gb = disk.free / (1024 ** 3)

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ops': {
            'all_sleeves_up': _proc_count('scripts/run_all_sleeves.py --with-aggressive-modes') > 0,
            'coinbase_loop_up': _proc_count(
                'scripts/run_shadow_training_loop.py --broker coinbase',
                exclude=('--profile crypto_futures',),
            ) > 0,
            'coinbase_futures_loop_up': _proc_count(
                'scripts/run_shadow_training_loop.py --broker coinbase --profile crypto_futures'
            ) > 0,
            'sql_link_writer_up': (
                _proc_count('scripts/ops/sql_link_shard_manager.py')
                + _proc_count('scripts/ops/sql_link_writer_service.py')
            ) > 0,
        },
        'system': {
            'disk_free_gb': round(free_gb, 2),
        },
        'quality': {
            'data_quality_score': round(_to_float(one.get('data_quality_score', one.get('data_quality_score', 0.0)), 0.0), 3),
            'blocked_rate': round(_to_float(one.get('combined_blocked_rate', 0.0), 0.0), 6),
            'decision_stale_windows_4h': int(_to_float(one.get('decision_stale_windows_4h', 0.0), 0.0)),
        },
        'promotion': {
            'promote_ok': bool(readiness.get('promote_ok', False)),
            'fail_share': round(_to_float(readiness.get('fail_share', 1.0), 1.0), 6),
            'considered_bots': int(_to_float(readiness.get('considered_bots', 0), 0)),
            'failed_bots': int(_to_float(readiness.get('failed_bots', 0), 0)),
        },
        'verify': {
            'ok': bool(verify.get('ok', False)),
            'failed_checks': verify.get('failed_checks', []) if isinstance(verify.get('failed_checks'), list) else [],
        },
        'source_verification': {
            'all_verified': bool(((source_verification.get('overall') or {}).get('all_verified', False))),
            'cross_verified_count': int(((((source_verification.get('overall') or {}).get('counts') or {}).get('cross_verified', 0)) or 0)),
            'single_source_verified_count': int(((((source_verification.get('overall') or {}).get('counts') or {}).get('single_source_verified', 0)) or 0)),
            'single_source_unverified_count': int(((((source_verification.get('overall') or {}).get('counts') or {}).get('single_source_unverified', 0)) or 0)),
            'unverified_sources': ((source_verification.get('overall') or {}).get('unverified_sources', []) if isinstance((source_verification.get('overall') or {}).get('unverified_sources'), list) else []),
        },
        'sleeves': {
            'dividend': {
                'loop_state': str(dividend_ingress.get('loop_state') or 'unknown'),
                'pause_reason': str(dividend_ingress.get('pause_reason') or ''),
                'paper_executions': int(_to_float(dividend_latest.get('executions', 0), 0)),
                'ending_net_pnl_total': round(_to_float(dividend_latest.get('ending_net_pnl_total', 0.0), 0.0), 6),
                'ending_realized_pnl_total': round(_to_float(dividend_latest.get('ending_realized_pnl_total', 0.0), 0.0), 6),
                'ending_unrealized_pnl_total': round(_to_float(dividend_latest.get('ending_unrealized_pnl_total', 0.0), 0.0), 6),
            },
        },
    }

    day = datetime.now(timezone.utc).strftime('%Y%m%d')
    out_dir = PROJECT_ROOT / 'exports' / 'reports'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'daily_ops_report_{day}.json'
    out_md = out_dir / f'daily_ops_report_{day}.md'
    latest_json = out_dir / 'daily_ops_report_latest.json'
    latest_md = out_dir / 'daily_ops_report_latest.md'

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
    latest_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    lines = [
        f"# Daily Ops Report ({payload['timestamp_utc']})",
        f"- all_sleeves_up: {payload['ops']['all_sleeves_up']}",
        f"- coinbase_loop_up: {payload['ops']['coinbase_loop_up']}",
        f"- coinbase_futures_loop_up: {payload['ops']['coinbase_futures_loop_up']}",
        f"- sql_link_writer_up: {payload['ops']['sql_link_writer_up']}",
        f"- disk_free_gb: {payload['system']['disk_free_gb']}",
        f"- data_quality_score: {payload['quality']['data_quality_score']}",
        f"- blocked_rate: {payload['quality']['blocked_rate']}",
        f"- decision_stale_windows_4h: {payload['quality']['decision_stale_windows_4h']}",
        f"- promote_ok: {payload['promotion']['promote_ok']}",
        f"- fail_share: {payload['promotion']['fail_share']}",
        f"- verify_ok: {payload['verify']['ok']}",
        f"- failed_checks: {','.join(payload['verify']['failed_checks']) if payload['verify']['failed_checks'] else 'none'}",
        f"- source_verification_all_verified: {payload['source_verification']['all_verified']}",
        f"- source_verification_cross_verified_count: {payload['source_verification']['cross_verified_count']}",
        f"- source_verification_single_source_verified_count: {payload['source_verification']['single_source_verified_count']}",
        f"- source_verification_single_source_unverified_count: {payload['source_verification']['single_source_unverified_count']}",
        f"- source_verification_unverified_sources: {','.join(payload['source_verification']['unverified_sources']) if payload['source_verification']['unverified_sources'] else 'none'}",
        f"- dividend_loop_state: {payload['sleeves']['dividend']['loop_state']}",
        f"- dividend_pause_reason: {payload['sleeves']['dividend']['pause_reason'] or 'none'}",
        f"- dividend_paper_executions: {payload['sleeves']['dividend']['paper_executions']}",
        f"- dividend_ending_net_pnl_total: {payload['sleeves']['dividend']['ending_net_pnl_total']}",
    ]
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    latest_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    webhook = os.getenv('OPS_REPORT_WEBHOOK_URL', '').strip()
    webhook_ok = False
    webhook_err = ''
    if webhook:
        try:
            _post_webhook(webhook, payload)
            webhook_ok = True
        except Exception as exc:
            webhook_err = str(exc)

    payload['webhook'] = {'configured': bool(webhook), 'ok': webhook_ok, 'error': webhook_err}
    latest_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"daily_ops_report ok verify={payload['verify']['ok']} promote_ok={payload['promotion']['promote_ok']}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
