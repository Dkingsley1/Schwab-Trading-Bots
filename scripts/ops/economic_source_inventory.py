#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.economic_source_registry import build_economic_source_inventory, load_economic_source_registry
from scripts.ops.long_runtime_common import write_payload


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "governance" / "health" / "economic_source_inventory_latest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and list routed public macro and microeconomic sources.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print one source per line instead of the compact summary.")
    args = parser.parse_args()
    inventory = build_economic_source_inventory(load_economic_source_registry(), project_root=PROJECT_ROOT)
    inventory["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    write_payload(args.output, inventory)
    if args.list:
        for row in inventory["sources"]:
            print(
                "{source_id}\t{granularity}\t{producer}\t{publisher}".format(
                    source_id=row.get("source_id", ""),
                    granularity="+".join(row.get("granularity") or []),
                    producer=row.get("producer_id", ""),
                    publisher=row.get("publisher", ""),
                )
            )
    elif args.json:
        print(json.dumps(inventory, ensure_ascii=True))
    else:
        summary = inventory["summary"]
        print(
            "economic_source_inventory status={status} routed={count} direct={direct} grouped={grouped} "
            "macro={macro} micro={micro} both={both}".format(
                status="ready" if inventory["validation"]["ok"] else "failed",
                count=summary["total_routed_source_count"],
                direct=summary["direct_source_count"],
                grouped=summary["grouped_source_count"],
                macro=summary["granularity_counts"].get("macro", 0),
                micro=summary["granularity_counts"].get("micro", 0),
                both=summary["granularity_counts"].get("both", 0),
            )
        )
    return 0 if inventory["validation"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
