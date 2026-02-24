import argparse
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / '.venv312' / 'bin' / 'python'
SHADOW_LOOP = PROJECT_ROOT / 'scripts' / 'run_shadow_training_loop.py'

DEFAULT_DIVIDEND_SYMBOLS = (
    'SCHD,VIG,DGRO,HDV,NOBL,SPYD,DIV,'
    'JNJ,PG,KO,PEP,MCD,XOM,CVX,ABT,MRK,PFE,T,VZ,O,MAIN'
)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run dedicated dividend shadow masterbot profile.')
    parser.add_argument('--broker', default=os.getenv('DATA_BROKER', 'schwab'), choices=['schwab', 'coinbase'])
    parser.add_argument('--simulate', action='store_true', help='Use simulated market feed.')
    parser.add_argument('--symbols', default=os.getenv('DIVIDEND_SYMBOLS', DEFAULT_DIVIDEND_SYMBOLS))
    parser.add_argument('--interval-seconds', type=int, default=int(os.getenv('DIVIDEND_SHADOW_INTERVAL', '60')))
    parser.add_argument('--max-iterations', type=int, default=int(os.getenv('DIVIDEND_SHADOW_MAX_ITERS', '0')))
    parser.add_argument('--auto-retrain', action='store_true', default=False)
    args = parser.parse_args()

    if not VENV_PY.exists():
        print(f'ERROR: missing venv python: {VENV_PY}')
        return 2
    if not SHADOW_LOOP.exists():
        print(f'ERROR: missing shadow loop script: {SHADOW_LOOP}')
        return 2

    env = os.environ.copy()
    env['MARKET_DATA_ONLY'] = '1'
    env['ALLOW_ORDER_EXECUTION'] = '0'
    env['SHADOW_PROFILE'] = 'dividend'
    env['SHADOW_DOMAIN'] = 'equities'
    env.setdefault('SHADOW_THRESHOLD_SHIFT', '+0.03')

    cmd = [
        str(VENV_PY),
        str(SHADOW_LOOP),
        '--broker',
        args.broker,
        '--symbols',
        args.symbols,
        '--interval-seconds',
        str(args.interval_seconds),
        '--max-iterations',
        str(args.max_iterations),
    ]
    if args.simulate:
        cmd.append('--simulate')
    if args.auto_retrain:
        cmd.append('--auto-retrain')

    print('Starting dividend shadow profile...')
    print('Symbols:', args.symbols)
    print('Command:', ' '.join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.wait()


if __name__ == '__main__':
    raise SystemExit(main())
