import argparse
import csv
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]




def _resolve_latest_path(pattern: str) -> Path | None:
    files = [Path(x) for x in glob.glob(pattern)]
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)


def _resolve_input_path(primary: Path, fallback_pattern: str) -> Path:
    if primary.exists():
        return primary
    fallback = _resolve_latest_path(fallback_pattern)
    return fallback or primary

def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: List[Dict], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def _decision_rows(src_rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for o in src_rows:
        out.append(
            {
                "timestamp_utc": o.get("timestamp_utc"),
                "symbol": o.get("symbol"),
                "status": o.get("status"),
                "strategy": o.get("strategy"),
                "action": o.get("action"),
                "quantity": o.get("quantity"),
                "model_score": o.get("model_score"),
                "threshold": o.get("threshold"),
                "reasons": "; ".join(o.get("reasons", [])),
                "gates": json.dumps(o.get("gates", {}), ensure_ascii=True),
                "safety": json.dumps(o.get("safety", {}), ensure_ascii=True),
                "metadata": json.dumps(o.get("metadata", {}), ensure_ascii=True),
            }
        )
    return out


def _governance_rows(src_rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for o in src_rows:
        out.append(
            {
                "timestamp_utc": o.get("timestamp_utc"),
                "symbol": o.get("symbol"),
                "snapshot_id": o.get("snapshot_id"),
                "master_action": o.get("master_action"),
                "master_score": o.get("master_score"),
                "master_vote": o.get("master_vote"),
                "active_sub_bots": o.get("active_sub_bots"),
                "inactive_sub_bots": o.get("inactive_sub_bots"),
                "options_style": (o.get("options_plan", {}) or {}).get("options_style"),
                "options_contracts": (o.get("options_plan", {}) or {}).get("contracts"),
                "market": json.dumps(o.get("market", {}), ensure_ascii=True),
                "context_market": json.dumps(o.get("context_market", {}), ensure_ascii=True),
                "master_outputs": json.dumps(o.get("master_outputs", {}), ensure_ascii=True),
                "grand_master_weights": json.dumps(o.get("grand_master_weights", {}), ensure_ascii=True),
                "options_master": json.dumps(o.get("options_master", {}), ensure_ascii=True),
                "recommendations": json.dumps(o.get("recommendations", []), ensure_ascii=True),
            }
        )
    return out


def _publish_latest_alias(out_dir: Path, named_file: Path, alias_name: str) -> None:
    alias_path = out_dir / alias_name
    if not named_file.exists():
        return
    alias_path.write_bytes(named_file.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export shadow decision/governance JSONL logs to CSV.")
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y%m%d"), help="UTC date in YYYYMMDD")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "exports" / "csv"),
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--latest-aliases",
        action="store_true",
        help="Also publish stable latest_decision_explanations.csv and latest_master_control.csv",
    )
    args = parser.parse_args()

    day = args.date
    out_dir = Path(args.out_dir)

    decision_path = _resolve_input_path(
        PROJECT_ROOT / "decision_explanations" / "shadow" / f"decision_explanations_{day}.jsonl",
        str(PROJECT_ROOT / "decision_explanations" / "shadow*" / f"decision_explanations_{day}.jsonl"),
    )
    governance_path = _resolve_input_path(
        PROJECT_ROOT / "governance" / "shadow" / f"master_control_{day}.jsonl",
        str(PROJECT_ROOT / "governance" / "shadow*" / f"master_control_{day}.jsonl"),
    )

    decision_rows = _decision_rows(_load_jsonl(decision_path))
    governance_rows = _governance_rows(_load_jsonl(governance_path))

    decision_csv = out_dir / f"decision_explanations_{day}.csv"
    governance_csv = out_dir / f"master_control_{day}.csv"

    if decision_rows:
        _write_csv(decision_csv, decision_rows, decision_rows[0].keys())
        print(f"Wrote {decision_csv} ({len(decision_rows)} rows)")
        if args.latest_aliases:
            _publish_latest_alias(out_dir, decision_csv, "latest_decision_explanations.csv")
            print(f"Wrote {out_dir / 'latest_decision_explanations.csv'}")
    else:
        print(f"No decision rows found for {day}: {decision_path}")

    if governance_rows:
        _write_csv(governance_csv, governance_rows, governance_rows[0].keys())
        print(f"Wrote {governance_csv} ({len(governance_rows)} rows)")
        if args.latest_aliases:
            _publish_latest_alias(out_dir, governance_csv, "latest_master_control.csv")
            print(f"Wrote {out_dir / 'latest_master_control.csv'}")
    else:
        print(f"No governance rows found for {day}: {governance_path}")


if __name__ == "__main__":
    main()
