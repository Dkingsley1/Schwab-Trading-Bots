import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe checkpoint/restore readiness.")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "governance" / "shadow_conservative_equities" / "runtime_checkpoint.json"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "health" / "restore_probe_latest.json"))
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        cands = sorted(PROJECT_ROOT.glob("governance/shadow*/runtime_checkpoint.json"))
        if cands:
            ckpt = cands[0]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt),
        "exists": ckpt.exists(),
        "ok": False,
    }

    if ckpt.exists():
        obj = _read_json(ckpt)
        iter_count = int(obj.get("iter_count", 0) or 0)
        cfg_hash = str(obj.get("config_hash", ""))
        payload.update({
            "iter_count": iter_count,
            "config_hash": cfg_hash,
            "ok": iter_count >= 0 and bool(cfg_hash),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
