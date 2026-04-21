#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "decision_provenance_cards_latest.json"
DECISION_RE = re.compile(
    r"mode=(?P<mode>\S+)\s+status=(?P<status>\S+)\s+symbol=(?P<symbol>\S+)\s+action=(?P<action>\S+)\s+score=(?P<score>[-+]?\d*\.?\d+)\s+threshold=(?P<threshold>[-+]?\d*\.?\d+)"
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _last_nonempty_line(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    for raw in reversed(lines):
        if raw.strip():
            return raw.strip()
    return ""


def _extract_field(line: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=(\S+)", line)
    return str(match.group(1) if match else "").strip()


def _parse_reasons(line: str) -> list[str]:
    if "reasons=" not in line:
        return []
    tail = line.split("reasons=", 1)[1]
    tail = tail.split(" gates=", 1)[0]
    tail = tail.split(" safety=", 1)[0]
    return [part.strip() for part in tail.split("|") if part.strip()]


def _parse_card(path: Path) -> dict[str, Any] | None:
    line = _last_nonempty_line(path)
    if not line:
        return None
    match = DECISION_RE.search(line)
    if not match:
        return None
    score = _safe_float(match.group("score"), 0.0)
    threshold = _safe_float(match.group("threshold"), 0.0)
    reasons = _parse_reasons(line)
    gates = ""
    if " gates=" in line:
        gates = line.split(" gates=", 1)[1].split(" safety=", 1)[0].strip()
    safety = line.split(" safety=", 1)[1].strip() if " safety=" in line else ""
    return {
        "mode": match.group("mode"),
        "status": match.group("status"),
        "symbol": match.group("symbol"),
        "action": match.group("action"),
        "score": round(score, 6),
        "threshold": round(threshold, 6),
        "edge": round(score - threshold, 6),
        "bot_id": _extract_field(line, "bot_id"),
        "bot_role": _extract_field(line, "bot_role"),
        "thesis_tokens": reasons[:4],
        "gates_excerpt": gates,
        "safety_excerpt": safety,
        "source_path": str(path),
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, limit: int = 8) -> dict[str, Any]:
    root = project_root / "decision_explanations"
    cards = []
    for path in sorted(root.glob("shadow*/latest_decisions.log")):
        card = _parse_card(path)
        if card is not None:
            cards.append(card)
    cards.sort(key=lambda row: abs(_safe_float(row.get("edge"), 0.0)), reverse=True)
    cards = cards[: max(int(limit), 1)]

    review_seed = json.dumps(cards, ensure_ascii=True, sort_keys=True)
    review_sha256 = hashlib.sha256(review_seed.encode("utf-8")).hexdigest()
    overall_status = "ready" if cards else "degraded"
    recommended_actions = ordered_unique(
        [
            "restore recent decision explanation logs if provenance cards are empty" if not cards else "",
            "use the cards to explain the last symbol-level action before changing thresholds or overrides" if cards else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(cards),
        "overall_status": overall_status,
        "card_count": len(cards),
        "mode_count": len({str(card.get("mode") or "") for card in cards}),
        "review_sha256": review_sha256,
        "recent_cards": cards,
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render compact decision provenance cards from the latest explanation logs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), limit=int(args.limit))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "decision_provenance_cards "
            f"overall_status={payload.get('overall_status', '')} "
            f"card_count={int(payload.get('card_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
