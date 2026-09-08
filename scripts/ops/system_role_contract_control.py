#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.system_role_contracts import build_contract_report, evaluate_component_action
from scripts.ops.long_runtime_common import write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_role_contract_latest.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate executable component roles, authority, and single-writer state ownership."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ownership-file", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--component-id", "--component", dest="component_id", default="")
    parser.add_argument("--action", default="")
    parser.add_argument("--state-domain", default="")
    parser.add_argument("--resource-path", "--resource", dest="resource_path", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if args.component_id or args.action:
        decision = evaluate_component_action(
            project_root,
            component_id=args.component_id,
            action=args.action,
            state_domain=args.state_domain,
            resource_path=args.resource_path,
            config_path=args.config,
        )
        if args.json:
            print(json.dumps(decision, ensure_ascii=True))
        else:
            print(
                "system_role_authority "
                f"ok={str(decision['ok']).lower()} "
                f"component={decision['component_id'] or 'missing'} "
                f"action={decision['action'] or 'missing'}"
            )
        return 0 if decision["ok"] else 2

    payload = build_contract_report(
        project_root,
        config_path=args.config,
        ownership_path=args.ownership_file,
        registry_path=args.registry,
    )
    out_path = args.out_file or project_root / "governance" / "health" / DEFAULT_OUT_PATH.name
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload["summary"]
        print(
            "system_role_contract "
            f"status={payload['overall_status']} grade={payload['grade']} "
            f"roles={summary['role_count']} components={summary['component_count']} "
            f"domains={summary['state_domain_count']} conflicts={summary['authority_conflict_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
