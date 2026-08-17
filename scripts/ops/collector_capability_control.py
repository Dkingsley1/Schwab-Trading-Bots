#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.collector_capability_routing import build_capability_routing
    from scripts.ops.long_runtime_common import load_json, write_payload
else:
    from core.collector_capability_routing import build_capability_routing
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "collector_capability_catalog_v1.json"
DEFAULT_COLLECTOR_CONTRACTS_PATH = PROJECT_ROOT / "governance" / "health" / "collector_contracts_latest.json"
DEFAULT_HIERARCHY_PATH = PROJECT_ROOT / "governance" / "bot_organization" / "bot_hierarchy_latest.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "collector_capability_control_latest.json"
DEFAULT_ROUTING_OUT_PATH = (
    PROJECT_ROOT / "governance" / "collector_capabilities" / "bot_subscriptions_latest.json"
)


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    collector_contracts_path: Path | None = None,
    hierarchy_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = project_root.resolve()
    config_path = config_path or root / "config" / DEFAULT_CONFIG_PATH.name
    collector_contracts_path = (
        collector_contracts_path
        or root / "governance" / "health" / DEFAULT_COLLECTOR_CONTRACTS_PATH.name
    )
    hierarchy_path = hierarchy_path or root / "governance" / "bot_organization" / DEFAULT_HIERARCHY_PATH.name
    return build_capability_routing(
        root,
        load_json(config_path),
        load_json(collector_contracts_path),
        load_json(hierarchy_path),
    )


def _resolve(root: Path, value: str | None, default: Path) -> Path:
    if not value:
        return root / default.relative_to(PROJECT_ROOT)
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the execution-free collector capability catalog health and shared bot subscriptions."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config")
    parser.add_argument("--collector-contracts")
    parser.add_argument("--hierarchy")
    parser.add_argument("--out-file")
    parser.add_argument("--routing-out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    config_path = _resolve(root, args.config, DEFAULT_CONFIG_PATH)
    contracts_path = _resolve(root, args.collector_contracts, DEFAULT_COLLECTOR_CONTRACTS_PATH)
    hierarchy_path = _resolve(root, args.hierarchy, DEFAULT_HIERARCHY_PATH)
    out_path = _resolve(root, args.out_file, DEFAULT_OUT_PATH)
    routing_out = _resolve(root, args.routing_out, DEFAULT_ROUTING_OUT_PATH)

    health, routing = build_payload(
        root,
        config_path=config_path,
        collector_contracts_path=contracts_path,
        hierarchy_path=hierarchy_path,
    )
    health["routing_artifact"] = str(routing_out.relative_to(root)) if routing_out.is_relative_to(root) else str(routing_out)
    write_payload(routing_out, routing)
    write_payload(out_path, health)

    if args.json:
        print(json.dumps(health, ensure_ascii=True, sort_keys=True))
    else:
        summary = health.get("summary") if isinstance(health.get("summary"), dict) else {}
        print(
            "collector_capability_control "
            f"status={health.get('overall_status')} paper_soak_ready={str(bool(health.get('paper_soak_ready'))).lower()} "
            f"planes={summary.get('plane_count', 0)} capabilities={summary.get('capability_count', 0)} "
            f"bots={summary.get('bot_binding_count', 0)}/{summary.get('assignment_count', 0)} "
            f"profiles={summary.get('subscription_profile_count', 0)}"
        )
    return 0 if bool(health.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
