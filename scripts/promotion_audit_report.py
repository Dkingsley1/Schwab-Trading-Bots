import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate promotion/hold audit report from master registry.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--out-json", default=str(PROJECT_ROOT / "governance" / "audits" / "promotion_audit_latest.json"))
    parser.add_argument("--out-md", default=str(PROJECT_ROOT / "governance" / "audits" / "promotion_audit_latest.md"))
    args = parser.parse_args()

    reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    rows = reg.get("sub_bots", [])

    promoted = [r for r in rows if r.get("promoted")]
    held = [r for r in rows if not r.get("promoted")]

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(rows),
            "promoted": len(promoted),
            "held": len(held),
        },
        "promoted": promoted,
        "held": held,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Promotion Audit ({payload['timestamp_utc']})",
        "",
        f"- Total bots: {len(rows)}",
        f"- Promoted: {len(promoted)}",
        f"- Held: {len(held)}",
        "",
        "## Top Active",
    ]
    for row in reg.get("summary", {}).get("top_active", []):
        lines.append(f"- {row.get('bot_id')} [{row.get('bot_role','unknown')}] weight={row.get('weight')} acc={row.get('test_accuracy')}")

    lines.append("")
    lines.append("## Promoted")
    for row in promoted[:30]:
        lines.append(f"- {row.get('bot_id')} role={row.get('bot_role','unknown')} reason={row.get('promotion_reason')}")

    lines.append("")
    lines.append("## Held")
    for row in held[:30]:
        lines.append(f"- {row.get('bot_id')} role={row.get('bot_role','unknown')} reason={row.get('promotion_reason')}")

    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
