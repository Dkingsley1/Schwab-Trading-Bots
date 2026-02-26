import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description='Re-evaluate storage route and auto-sync local backlog when drive is back.')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    from core.storage_router import describe_storage_routing, route_runtime_storage

    routing = route_runtime_storage(PROJECT_ROOT)

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'mode': routing.mode,
        'active_root': str(routing.active_root),
        'switched_links': list(routing.switched_links),
        'passthrough_paths': list(routing.passthrough_paths),
        'autosync': {
            'copied_files': int(routing.autosync_copied_files),
            'copy_errors': int(routing.autosync_copy_errors),
            'pruned_files': int(routing.autosync_pruned_files),
        },
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'storage_failback_sync_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(describe_storage_routing(routing))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
