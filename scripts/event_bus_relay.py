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


def _runtime_sources(project_root: Path, day: str) -> list[Path]:
    channel_sources = sorted(
        p for p in (project_root / "governance" / "channels" / "runtime").glob(f"*/runtime_{day}.jsonl") if p.is_file()
    )
    if channel_sources:
        return channel_sources
    legacy = project_root / "governance" / "events" / f"runtime_events_{day}.jsonl"
    return [legacy] if legacy.is_file() else []


def relay_day(project_root: Path, *, day: str, state_path: Path) -> dict[str, int | str]:
    sources = _runtime_sources(project_root, day)
    if not sources:
        return {"processed": 0, "source_count": 0, "last_line": 0, "mode": "missing"}

    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    out_root = project_root / "governance" / "events" / "consumers"
    out_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    latest_line = 0
    for src in sources:
        last = int(state.get(str(src), 0) or 0)
        newest = last
        for line_no, raw in _iter_new_lines(src, last):
            newest = line_no
            latest_line = max(latest_line, line_no)
            processed += 1
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            event = str(obj.get("event", "unknown"))
            out = out_root / f"{event}_{day}.jsonl"
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=True) + "\n")
        state[str(src)] = newest

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "processed": processed,
        "source_count": len(sources),
        "last_line": latest_line,
        "mode": "channels" if any("/governance/channels/runtime/" in str(src).replace("\\", "/") for src in sources) else "legacy",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Relay runtime event bus JSONL to partitioned consumer streams.")
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--state-file", default=str(PROJECT_ROOT / "governance" / "events" / "relay_state.json"))
    args = parser.parse_args()
    state_path = Path(args.state_file)
    payload = relay_day(PROJECT_ROOT, day=args.day, state_path=state_path)
    if int(payload.get("source_count", 0) or 0) <= 0:
        print(f"no_source_day={args.day}")
        return 0
    print(
        f"relay_processed={int(payload.get('processed', 0) or 0)} "
        f"sources={int(payload.get('source_count', 0) or 0)} "
        f"mode={payload.get('mode', 'missing')} "
        f"last_line={int(payload.get('last_line', 0) or 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
