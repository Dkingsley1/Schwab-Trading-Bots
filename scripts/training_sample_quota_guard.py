import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Guard retrain against regime/symbol sample concentration.")
    ap.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "trade_history" / "trade_learning_dataset.json"))
    ap.add_argument("--min-per-regime", type=int, default=120)
    ap.add_argument("--min-per-symbol", type=int, default=25)
    ap.add_argument("--max-top-symbol-share", type=float, default=0.25)
    ap.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "training_sample_quota_latest.json"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    ds = _load(Path(args.dataset))
    ex = ds.get("examples") if isinstance(ds.get("examples"), list) else []
    if not ex:
        ex = ds.get("data") if isinstance(ds.get("data"), list) else []

    regime = Counter()
    symbol = Counter()
    for row in ex:
        if not isinstance(row, dict):
            continue
        regime[str(row.get("regime", "other"))] += 1
        symbol[str(row.get("symbol", "")).upper()] += 1

    rows = sum(regime.values())
    failed = []
    regime_below = {k: int(v) for k, v in regime.items() if int(v) < int(args.min_per_regime)}
    if regime_below:
        failed.append("regime_quota")

    top_symbol = symbol.most_common(1)[0] if symbol else ("", 0)
    top_share = (float(top_symbol[1]) / float(rows)) if rows > 0 else 0.0
    if rows > 0 and top_share > float(args.max_top_symbol_share):
        failed.append("symbol_concentration")

    if rows > 0 and len(symbol) > 0:
        viable_symbols = sum(1 for _, c in symbol.items() if int(c) >= int(args.min_per_symbol))
        if viable_symbols <= 0:
            failed.append("symbol_quota")
    else:
        failed.append("empty_dataset")

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": len(failed) == 0,
        "failed_checks": failed,
        "rows": int(rows),
        "regime_counts": {k: int(v) for k, v in regime.items()},
        "symbol_count": int(len(symbol)),
        "top_symbol": {"symbol": str(top_symbol[0]), "count": int(top_symbol[1]), "share": round(float(top_share), 6)},
        "thresholds": {
            "min_per_regime": int(args.min_per_regime),
            "min_per_symbol": int(args.min_per_symbol),
            "max_top_symbol_share": float(args.max_top_symbol_share),
        },
        "regimes_below_quota": regime_below,
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(f"training_sample_quota_ok={int(out['ok'])} failed_checks={','.join(failed) if failed else 'none'}")

    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
