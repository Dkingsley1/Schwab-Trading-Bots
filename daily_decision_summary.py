import json
import os
from collections import Counter
from datetime import datetime, timezone


def _load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize(project_root: str, day: str):
    decisions_path = os.path.join(project_root, "decisions", f"trade_decisions_{day}.jsonl")
    paper_path = os.path.join(project_root, "paper_trades.jsonl")

    decisions = _load_jsonl(decisions_path)
    paper = _load_jsonl(paper_path)

    print(f"Date: {day}")
    print(f"Decisions file: {decisions_path}")
    print(f"Paper file: {paper_path}")

    if not decisions:
        print("No decisions logged for this day.")
        return

    total = len(decisions)
    decision_counter = Counter(d.get("decision", "UNKNOWN") for d in decisions)
    action_counter = Counter(d.get("action", "UNKNOWN") for d in decisions)
    mode_counter = Counter((d.get("metadata") or {}).get("mode", "unknown") for d in decisions)

    scores = [float(d.get("model_score", 0.0)) for d in decisions]
    avg_score = sum(scores) / max(len(scores), 1)

    print("\nDecision Summary")
    print(f"Total decisions: {total}")
    print(f"EXECUTE: {decision_counter.get('EXECUTE', 0)}")
    print(f"BLOCK: {decision_counter.get('BLOCK', 0)}")
    print(f"Average model_score: {avg_score:.4f}")

    print("\nBy Action")
    for k, v in sorted(action_counter.items()):
        print(f"{k}: {v}")

    print("\nBy Mode")
    for k, v in sorted(mode_counter.items()):
        print(f"{k}: {v}")

    # Paper summary limited to same UTC day.
    paper_today = [p for p in paper if str(p.get("timestamp_utc", "")).startswith(f"{day[0:4]}-{day[4:6]}-{day[6:8]}")]
    print("\nPaper Summary")
    print(f"Paper orders today: {len(paper_today)}")
    if paper_today:
        p_actions = Counter(p.get("action", "UNKNOWN") for p in paper_today)
        for k, v in sorted(p_actions.items()):
            print(f"{k}: {v}")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    day = os.getenv("SUMMARY_DAY", datetime.now(timezone.utc).strftime("%Y%m%d"))
    summarize(project_root, day)


if __name__ == "__main__":
    main()
