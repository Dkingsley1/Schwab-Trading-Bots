#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

from core.accountability import safe_write_json_atomic
from core.authoritative_systems import load_registry
from core.strategy_validity import (
    future_suffix_invariance,
    recursive_warmup_stability,
    scan_paths,
    validity_contract_receipt,
)


DEFAULT_OUT = Path("governance/health/strategy_validity_control_latest.json")


def _trailing_mean(values: Sequence[float]) -> list[float]:
    rows: list[float] = []
    for index in range(len(values)):
        window = values[max(0, index - 2) : index + 1]
        rows.append(sum(window) / len(window))
    return rows


def build_payload(
    project_root: Path = PROJECT_ROOT, *, paths: Sequence[Path] | None = None
) -> dict[str, Any]:
    if paths is None:
        registry = load_registry(
            project_root / "config" / "authoritative_systems_v1.json"
        )
        controls = dict(registry.get("controls") or {})
        owners = {
            str(row.get("owner") or "")
            for row in controls.values()
            if isinstance(row, dict) and str(row.get("owner") or "").endswith(".py")
        }
        validity = controls.get("point_in_time_validity")
        configured = {
            str(value)
            for value in (
                validity.get("scan_paths") if isinstance(validity, dict) else []
            )
            if str(value).endswith(".py")
        }
        paths = sorted(project_root / owner for owner in owners | configured)
    static = scan_paths(paths)
    values = [float(index) for index in range(1, 65)]
    suffix = future_suffix_invariance(_trailing_mean, values)
    recursive = recursive_warmup_stability(
        _trailing_mean,
        values,
        startup_lengths=(16, 32, 64),
        comparison_points=1,
    )
    from core.sleeve_strategy_specialization import materialize_strategy_library

    library = materialize_strategy_library(project_root=project_root)
    validity_ready = 0
    for row in library.values():
        contract = row.get("validity_contract")
        if not isinstance(contract, dict):
            continue
        receipt = str(contract.get("receipt_sha256") or "")
        material = {
            key: value for key, value in contract.items() if key != "receipt_sha256"
        }
        required = (
            "point_in_time_features_required",
            "future_suffix_invariance_required",
            "recursive_warmup_stability_required",
            "high_confidence_static_lookahead_scan_required",
            "late_data_quarantine_required",
        )
        if all(
            contract.get(key) is True for key in required
        ) and receipt == validity_contract_receipt(material):
            validity_ready += 1
    contract_coverage = {
        "strategy_count": len(library),
        "validity_ready_count": validity_ready,
        "complete": bool(library and validity_ready == len(library)),
    }
    ok = bool(
        static["ok"]
        and suffix["ok"]
        and recursive["ok"]
        and contract_coverage["complete"]
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": "ready" if ok else "blocked",
        "static_analysis": static,
        "future_suffix_invariance": suffix,
        "recursive_warmup_stability": recursive,
        "strategy_contract_coverage": contract_coverage,
        "failure_behavior": "block_candidate_promotion_and_live_execution",
        "authority": {
            "can_change_action": False,
            "can_change_quantity": False,
            "can_promote_live": False,
            "can_submit_live_order": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate point-in-time and recursive strategy semantics."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    paths = [Path(value).expanduser() for value in args.path] or None
    payload = build_payload(root, paths=paths)
    out = Path(args.out_file).expanduser()
    if not out.is_absolute():
        out = root / out
    safe_write_json_atomic(
        str(out), payload, project_root=str(root), source="strategy_validity_control"
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(
            f"strategy_validity status={payload['overall_status']} "
            f"static_issues={payload['static_analysis']['issue_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
