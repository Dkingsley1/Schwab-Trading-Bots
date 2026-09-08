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
    from core.accountability import safe_write_json_atomic
    from core.sleeve_scalability_selector import build_selector_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.accountability import safe_write_json_atomic
    from core.sleeve_scalability_selector import build_selector_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sleeve_scalability_selector_v1.json"
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "sleeve_scalability_selector_latest.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    account_policy_key: str = "",
    execution_route_id: str = "",
    capital_usd: float | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name
    policy = _load_json(config_path)
    inputs = policy.get("inputs") if isinstance(policy.get("inputs"), dict) else {}
    artifacts = {
        name: _load_json(_resolve(project_root, raw_path))
        for name, raw_path in inputs.items()
        if isinstance(raw_path, str)
    }
    payload = build_selector_payload(
        policy,
        artifacts.get("profitability_manifest", {}),
        artifacts.get("bot_hierarchy", {}),
        artifacts.get("live_canary_graduation", {}),
        artifacts.get("account_position_study", {}),
        artifacts.get("profitability_firewall", {}),
        artifacts.get("paper_execution_calibration", {}),
        account_policy_key=account_policy_key,
        execution_route_id=execution_route_id,
        capital_usd=capital_usd,
    )
    payload["source_files"] = {
        "config": str(config_path),
        **{
            name: str(_resolve(project_root, raw_path))
            for name, raw_path in inputs.items()
            if isinstance(raw_path, str)
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank evidence-qualified sleeves and propose a bounded advisory portfolio."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--account-policy-key", default="")
    parser.add_argument("--execution-route-id", default="")
    parser.add_argument("--capital-usd", type=float)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config or project_root / "config" / DEFAULT_CONFIG_PATH.name
    if not config_path.is_absolute():
        config_path = project_root / config_path
    out_path = args.out_file or (
        project_root / "governance" / "health" / DEFAULT_OUT_PATH.name
    )
    if not out_path.is_absolute():
        out_path = project_root / out_path
    payload = build_payload(
        project_root,
        config_path=config_path,
        account_policy_key=args.account_policy_key,
        execution_route_id=args.execution_route_id,
        capital_usd=args.capital_usd,
    )
    safe_write_json_atomic(
        str(out_path),
        payload,
        project_root=str(project_root),
        source="sleeve_scalability_selector",
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        selected = payload.get("selected_advisory_plan")
        selected_ids = (
            ",".join(selected.get("sleeve_ids") or [])
            if isinstance(selected, dict)
            else "none"
        )
        print(
            "sleeve_scalability_selector "
            f"status={payload.get('overall_status', '')} "
            f"research_eligible={summary.get('research_eligible_sleeve_count', 0)} "
            f"application_eligible={summary.get('application_eligible_sleeve_count', 0)} "
            f"selected={selected_ids}"
        )
    return 0 if payload.get("control_ready", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
