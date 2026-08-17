#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.capability_materialization import build_materialized_capabilities
    from scripts.ops.long_runtime_common import load_json, write_payload
else:
    from core.capability_materialization import build_materialized_capabilities
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload


DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "capability_materialization_v1.json"
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "collector_capabilities" / "materialized_capabilities_latest.json"
)


def _resolve(root: Path, value: str | None, default: Path) -> Path:
    if not value:
        return root / default.relative_to(PROJECT_ROOT)
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    policy_path: Path | None = None,
) -> dict:
    root = project_root.resolve()
    effective_policy_path = policy_path or root / "config" / DEFAULT_POLICY_PATH.name
    policy = load_json(effective_policy_path)
    derivative_path = root / str(
        policy.get("derivative_contract_master_path")
        or "config/derivatives_contract_master_v1.json"
    )
    return build_materialized_capabilities(
        root,
        policy,
        load_json(derivative_path),
        derivative_master_path=derivative_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize source-backed calendar, session, derivative, and stress capabilities."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--policy")
    parser.add_argument("--out-file")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    policy_path = _resolve(root, args.policy, DEFAULT_POLICY_PATH)
    out_path = _resolve(root, args.out_file, DEFAULT_OUT_PATH)
    payload = build_payload(root, policy_path=policy_path)
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(
            "capability_materialization_control "
            f"status={payload.get('overall_status')} "
            f"ready={payload.get('ready_capability_count', 0)}/{payload.get('capability_count', 0)} "
            f"live_promotion_ready={str(bool(payload.get('live_promotion_ready'))).lower()}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
