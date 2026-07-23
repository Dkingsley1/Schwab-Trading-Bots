import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python

PY = resolve_runtime_python(PROJECT_ROOT)
PROMOTION_GATE_PATH = PROJECT_ROOT / 'governance' / 'walk_forward' / 'promotion_gate_latest.json'
CANDIDATE_ADMISSION_PATH = PROJECT_ROOT / 'governance' / 'health' / 'new_bot_admission_guard_promotion_pipeline_latest.json'


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _promotion_candidate_ids(promotion_gate: dict) -> list[str]:
    ids = {
        str(raw or '').strip()
        for raw in (promotion_gate.get('considered_bot_ids') or [])
        if str(raw or '').strip()
    }
    for key in ('pass_examples', 'near_pass_examples', 'fail_examples'):
        rows = promotion_gate.get(key) if isinstance(promotion_gate.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            bot_id = str(row.get('bot_id') or '').strip()
            if bot_id:
                ids.add(bot_id)
    return sorted(ids)


def _candidate_scoped_admission_cmd(base_cmd: list[str]) -> tuple[list[str], list[str]]:
    candidate_ids = _promotion_candidate_ids(_load_json(PROMOTION_GATE_PATH))
    if not candidate_ids:
        return base_cmd, candidate_ids
    cmd = [
        str(PY),
        str(PROJECT_ROOT / 'scripts' / 'new_bot_admission_guard.py'),
        '--include-bot-ids',
        ','.join(candidate_ids),
        '--out-file',
        str(CANDIDATE_ADMISSION_PATH),
        '--json',
    ]
    return cmd, candidate_ids


def _run(step: str, cmd: list[str]) -> dict:
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    return {
        'step': step,
        'cmd': cmd,
        'rc': int(p.returncode),
        'ok': p.returncode == 0,
        'stdout_tail': '\n'.join((p.stdout or '').splitlines()[-40:]),
        'stderr_tail': '\n'.join((p.stderr or '').splitlines()[-40:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Run full promotion pipeline with one artifact.')
    parser.add_argument('--apply-retirement', action='store_true', default=True)
    parser.add_argument('--run-master-update', action='store_true', default=True)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    steps: list[tuple[str, list[str], bool]] = [
        ('walk_forward_validate', [str(PY), str(PROJECT_ROOT / 'scripts' / 'walk_forward_validate.py')], True),
        ('walk_forward_promotion_gate', [str(PY), str(PROJECT_ROOT / 'scripts' / 'walk_forward_promotion_gate.py')], False),
        ('lane_promotion_gate', [str(PY), str(PROJECT_ROOT / 'scripts' / 'lane_promotion_gate.py'), '--json'], False),
        ('promotion_readiness_summary', [str(PY), str(PROJECT_ROOT / 'scripts' / 'promotion_readiness_summary.py'), '--json'], True),
        ('promotion_bottleneck_focus', [str(PY), str(PROJECT_ROOT / 'scripts' / 'promotion_bottleneck_focus.py'), '--json'], True),
        ('schema_migration_guard', [str(PY), str(PROJECT_ROOT / 'scripts' / 'schema_migration_guard.py'), '--json'], True),
        ('bot_support_owner_guard', [str(PY), str(PROJECT_ROOT / 'scripts' / 'bot_support_owner_guard.py'), '--json'], True),
        ('feature_store_manifest', [str(PY), str(PROJECT_ROOT / 'scripts' / 'feature_store_manifest.py'), '--json'], True),
        ('new_bot_graduation_gate', [str(PY), str(PROJECT_ROOT / 'scripts' / 'new_bot_graduation_gate.py'), '--json'], False),
        ('new_bot_admission_guard', [str(PY), str(PROJECT_ROOT / 'scripts' / 'new_bot_admission_guard.py'), '--json'], True),
        ('retrain_schema_compatibility_guard', [str(PY), str(PROJECT_ROOT / 'scripts' / 'retrain_schema_compatibility_guard.py'), '--json'], True),
        ('leak_overfit_guard', [str(PY), str(PROJECT_ROOT / 'scripts' / 'leak_overfit_guard.py'), '--json'], False),
        ('golden_replay_regression_guard', [str(PY), str(PROJECT_ROOT / 'scripts' / 'golden_replay_regression_guard.py'), '--json'], True),
        ('cohort_drift_baseline_guard', [str(PY), str(PROJECT_ROOT / 'scripts' / 'cohort_drift_baseline_guard.py'), '--json'], True),
        ('champion_challenger_probation_guard', [str(PY), str(PROJECT_ROOT / 'scripts' / 'champion_challenger_probation_guard.py'), '--json'], True),
        ('champion_challenger_probation_action', [str(PY), str(PROJECT_ROOT / 'scripts' / 'champion_challenger_probation_action.py'), '--json'], True),
        ('retrain_lane_scheduler', [str(PY), str(PROJECT_ROOT / 'scripts' / 'retrain_lane_scheduler.py'), '--json'], True),
        ('promotion_packet_builder', [str(PY), str(PROJECT_ROOT / 'scripts' / 'promotion_packet_builder.py'), '--json'], False),
        ('promotion_quality_gate', [str(PY), str(PROJECT_ROOT / 'scripts' / 'promotion_quality_gate.py'), '--json'], True),
    ]

    if args.apply_retirement:
        steps.append(
            ('retire_persistent_losers', [
                str(PY),
                str(PROJECT_ROOT / 'scripts' / 'retire_persistent_losers.py'),
                '--lookback-days', '14',
                '--min-fail-days', '5',
                '--min-no-improvement-streak', '2',
                '--max-retire-per-run', '6',
                '--apply',
                '--json',
            ], True)
        )

    if args.run_master_update:
        steps.append(('run_master_bot', [str(PY), str(PROJECT_ROOT / 'scripts' / 'run_master_bot.py'), '--require-canary-gate'], False))

    results = []
    hard_fail = False
    promotion_candidate_ids: list[str] = []
    candidate_scoped_new_bot_admission = False
    for step, cmd, required_zero in steps:
        if step == 'new_bot_admission_guard':
            cmd, promotion_candidate_ids = _candidate_scoped_admission_cmd(cmd)
            candidate_scoped_new_bot_admission = bool(promotion_candidate_ids)
        row = _run(step, cmd)
        if step == 'new_bot_admission_guard':
            row['promotion_candidate_ids'] = promotion_candidate_ids
            row['candidate_scoped'] = candidate_scoped_new_bot_admission
            row['candidate_scoped_out_file'] = str(CANDIDATE_ADMISSION_PATH) if candidate_scoped_new_bot_admission else ''
        results.append(row)
        if required_zero and row['rc'] != 0:
            hard_fail = True

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ok': not hard_fail,
        'hard_fail': hard_fail,
        'promotion_candidate_ids': promotion_candidate_ids,
        'candidate_scoped_new_bot_admission': candidate_scoped_new_bot_admission,
        'candidate_scoped_new_bot_admission_file': str(CANDIDATE_ADMISSION_PATH) if candidate_scoped_new_bot_admission else '',
        'steps': results,
    }

    out_latest = PROJECT_ROOT / 'governance' / 'walk_forward' / 'promotion_pipeline_latest.json'
    out_hist = PROJECT_ROOT / 'governance' / 'walk_forward' / 'promotion_pipeline_history.jsonl'
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out_latest.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
    with out_hist.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=True) + '\n')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"promotion_pipeline ok={payload['ok']} hard_fail={payload['hard_fail']}")

    return 0 if payload['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
