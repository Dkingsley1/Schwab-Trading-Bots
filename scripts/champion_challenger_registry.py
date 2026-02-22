import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _approval_meta(path: Path) -> tuple[bool, dict]:
    if not path.exists():
        return False, {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return False, {"error": "empty_approval_file"}
        payload = json.loads(text)
        ok = bool(payload.get("approved_by")) and bool(payload.get("approved_at_utc")) and bool(payload.get("ticket"))
        return ok, payload
    except Exception:
        return False, {"error": "approval_file_not_json"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Champion/challenger registry with signed promotion approval.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "governance" / "champion_challenger" / "registry.json"))
    parser.add_argument("--approval-file", default=str(PROJECT_ROOT / "governance" / "champion_challenger" / "PROMOTION_APPROVED.flag"))
    parser.add_argument("--candidate", default="walk_forward_candidate")
    args = parser.parse_args()

    gate = _load(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json")
    promote_ok = bool(gate.get("promote_ok", False))
    approved, approval_meta = _approval_meta(Path(args.approval_file))

    reg_path = Path(args.registry)
    reg = _load(reg_path)
    reg.setdefault("champion", {"name": "current", "since_utc": datetime.now(timezone.utc).isoformat()})
    reg.setdefault("history", [])

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": args.candidate,
        "promote_ok": promote_ok,
        "approval_present": Path(args.approval_file).exists(),
        "approval_valid": approved,
        "approval_meta": approval_meta,
        "action": "hold",
    }

    if promote_ok and approved:
        prev = reg.get("champion", {})
        reg["history"].append(prev)
        reg["champion"] = {
            "name": args.candidate,
            "since_utc": event["timestamp_utc"],
            "approved_by": approval_meta.get("approved_by"),
            "ticket": approval_meta.get("ticket"),
        }
        event["action"] = "promoted"

    reg["last_event"] = event
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(reg, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(event, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
