import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _iter_new_lines(path: Path, start: int):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i <= start:
                continue
            yield i, line.rstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Relay runtime event bus JSONL to partitioned consumer streams.")
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--state-file", default=str(PROJECT_ROOT / "governance" / "events" / "relay_state.json"))
    args = parser.parse_args()

    src = PROJECT_ROOT / "governance" / "events" / f"runtime_events_{args.day}.jsonl"
    if not src.exists():
        print(f"no_source={src}")
        return 0

    state_path = Path(args.state_file)
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    last = int(state.get(str(src), 0) or 0)

    out_root = PROJECT_ROOT / "governance" / "events" / "consumers"
    out_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    newest = last
    for line_no, raw in _iter_new_lines(src, last):
        newest = line_no
        processed += 1
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        event = str(obj.get("event", "unknown"))
        out = out_root / f"{event}_{args.day}.jsonl"
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")

    state[str(src)] = newest
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"relay_processed={processed} last_line={newest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
