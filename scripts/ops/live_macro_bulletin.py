#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "external_context" / "live_macro_latest.json"
DEFAULT_EVENT_DIR = PROJECT_ROOT / "governance" / "events"
FALLBACK_EVENT_PATH = Path("/tmp/live_macro_events.jsonl")

_POWELL_DEFAULT_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "IEF",
    "GLD",
    "UUP",
    "BTC-USD",
    "ETH-USD",
    "COIN",
    "MSTR",
    "TSLA",
    "NVDA",
]

_STANCE_HINTS = {
    "hawkish": -0.75,
    "dovish": 0.75,
    "neutral": 0.0,
    "mixed": -0.20,
}

_IMPACT_HINTS = {
    "low": 0.35,
    "medium": 0.60,
    "high": 0.85,
    "critical": 1.0,
}

_HAWKISH_TOKENS = (
    "higher for longer",
    "restrictive",
    "inflation remains too high",
    "upside risk",
    "no rush to cut",
    "tightening",
    "sticky inflation",
    "elevated inflation",
    "rates need to stay high",
    "not cutting",
)

_DOVISH_TOKENS = (
    "cuts",
    "cut rates",
    "easing",
    "disinflation",
    "cooling inflation",
    "support growth",
    "downside risk",
    "softening labor",
    "lower rates",
    "accommodative",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    for token in str(raw or "").replace("|", ",").split(","):
        symbol = token.strip().upper()
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _parse_timestamp(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return _now_iso()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception as exc:
        raise SystemExit(f"invalid --published value: {text} ({exc})")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _infer_stance(text: str) -> Tuple[str, float]:
    lowered = str(text or "").lower()
    hawkish_hits = sum(1 for token in _HAWKISH_TOKENS if token in lowered)
    dovish_hits = sum(1 for token in _DOVISH_TOKENS if token in lowered)
    if hawkish_hits > dovish_hits and hawkish_hits > 0:
        return "hawkish", _STANCE_HINTS["hawkish"]
    if dovish_hits > hawkish_hits and dovish_hits > 0:
        return "dovish", _STANCE_HINTS["dovish"]
    return "neutral", _STANCE_HINTS["neutral"]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, fallback: Path, row: Dict[str, Any]) -> str:
    encoded = json.dumps(row, ensure_ascii=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        return str(path)
    except Exception:
        fallback.parent.mkdir(parents=True, exist_ok=True)
        with fallback.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        return str(fallback)


def append_live_macro_event(
    *,
    event_type: str,
    payload: Dict[str, Any],
    out_file: Path,
    extra: Dict[str, Any] | None = None,
    fallback_path: Path = FALLBACK_EVENT_PATH,
) -> str:
    event_payload: Dict[str, Any] = {
        "timestamp_utc": _now_iso(),
        "event_type": str(event_type or "live_macro_event"),
        "category": "live_macro",
        "out_file": str(out_file),
        "payload": payload,
    }
    if extra:
        event_payload.update(extra)
    event_path = DEFAULT_EVENT_DIR / f"live_macro_events_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    return _append_jsonl(event_path, fallback_path, event_payload)


def build_live_macro_payload(
    *,
    template: str,
    headline: str,
    summary: str = "",
    content: str = "",
    speaker: str = "",
    source: str = "",
    url: str = "",
    symbols: str | List[str] = "",
    published: str = "",
    expires_hours: float = 4.0,
    stance: str = "auto",
    impact: str = "high",
    broad_market: bool | None = None,
    sentiment_hint_override: float | None = None,
    shock_hint_override: float | None = None,
    channel: str = "live_macro_bulletin",
) -> Dict[str, Any]:
    defaults = {
        "headline": "",
        "speaker": "",
        "source": "",
        "symbols": [],
        "broad_market": True,
        "shock_hint": _IMPACT_HINTS[str(impact or "high")],
    }
    if template == "powell":
        defaults.update(
            {
                "headline": "Jerome Powell live remarks",
                "speaker": "Jerome Powell",
                "source": "Federal Reserve",
                "symbols": list(_POWELL_DEFAULT_SYMBOLS),
                "broad_market": True,
                "shock_hint": max(_IMPACT_HINTS[str(impact or "high")], 0.90),
            }
        )
    elif template == "fed":
        defaults.update(
            {
                "headline": "Federal Reserve live macro event",
                "speaker": "Federal Reserve",
                "source": "Federal Reserve",
                "symbols": list(_POWELL_DEFAULT_SYMBOLS),
                "broad_market": True,
                "shock_hint": max(_IMPACT_HINTS[str(impact or "high")], 0.85),
            }
        )

    published_iso = _parse_timestamp(published)
    expires_at = datetime.fromisoformat(published_iso.replace("Z", "+00:00")) + timedelta(hours=max(float(expires_hours), 0.25))

    if isinstance(symbols, list):
        symbols_list = []
        for item in symbols:
            sym = str(item or "").strip().upper()
            if sym and sym not in symbols_list:
                symbols_list.append(sym)
    else:
        symbols_list = _parse_symbols(symbols)
    symbols_list = symbols_list or list(defaults["symbols"])

    headline_value = str(headline or defaults["headline"]).strip()
    summary_value = str(summary or "").strip()
    content_value = str(content or "").strip()
    speaker_value = str(speaker or defaults["speaker"]).strip()
    source_value = str(source or defaults["source"] or "manual_macro_bulletin").strip()
    url_value = str(url or "").strip()

    if not headline_value:
        raise SystemExit("headline is required unless a template supplies one")

    text_blob = " ".join(chunk for chunk in (headline_value, summary_value, content_value) if chunk).strip()
    if stance == "auto":
        stance_value, sentiment_hint = _infer_stance(text_blob)
    else:
        stance_value = str(stance)
        sentiment_hint = _STANCE_HINTS[stance_value]

    if sentiment_hint_override is not None:
        sentiment_hint = float(sentiment_hint_override)
    shock_hint = float(shock_hint_override) if shock_hint_override is not None else float(defaults["shock_hint"])
    broad_market_value = bool(defaults["broad_market"] if broad_market is None else broad_market)

    return {
        "timestamp_utc": _now_iso(),
        "category": "live_macro",
        "active": True,
        "template": template,
        "published": published_iso,
        "expires_at_utc": expires_at.astimezone(timezone.utc).isoformat(),
        "speaker": speaker_value,
        "source": source_value,
        "url": url_value,
        "broad_market": broad_market_value,
        "symbols": symbols_list,
        "stance": stance_value,
        "sentiment_hint": sentiment_hint,
        "shock_hint": shock_hint,
        "items": [
            {
                "headline": headline_value,
                "summary": summary_value,
                "content": content_value,
                "speaker": speaker_value,
                "source": source_value,
                "publisher": source_value,
                "channel": channel,
                "published": published_iso,
                "url": url_value,
                "symbols": symbols_list,
                "broad_market": broad_market_value,
                "macro_event": True,
                "topic": "macro",
                "stance": stance_value,
                "sentiment_hint": sentiment_hint,
                "shock_hint": shock_hint,
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish or clear a live macro bulletin for the shadow loops.")
    parser.add_argument("--template", choices=("powell", "fed", "generic"), default="generic")
    parser.add_argument("--headline", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--content", default="")
    parser.add_argument("--speaker", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--published", default="")
    parser.add_argument("--expires-hours", type=float, default=4.0)
    parser.add_argument("--stance", choices=("auto", "hawkish", "dovish", "neutral", "mixed"), default="auto")
    parser.add_argument("--impact", choices=("low", "medium", "high", "critical"), default="high")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out_file).expanduser().resolve()

    if args.status:
        if out_path.exists():
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            payload = {"active": False, "path": str(out_path)}
        print(json.dumps(payload, ensure_ascii=True, indent=2) if args.json else json.dumps(payload, ensure_ascii=True))
        return 0

    if args.clear:
        if out_path.exists():
            out_path.unlink()
        payload = {"timestamp_utc": _now_iso(), "active": False, "path": str(out_path), "cleared": True}
        events_file = append_live_macro_event(event_type="clear", payload=payload, out_file=out_path)
        payload["events_file"] = events_file
        print(json.dumps(payload, ensure_ascii=True, indent=2) if args.json else json.dumps(payload, ensure_ascii=True))
        return 0

    payload = build_live_macro_payload(
        template=args.template,
        headline=args.headline,
        summary=args.summary,
        content=args.content,
        speaker=args.speaker,
        source=args.source,
        url=args.url,
        symbols=args.symbols,
        published=args.published,
        expires_hours=args.expires_hours,
        stance=args.stance,
        impact=args.impact,
    )
    _write_json(out_path, payload)
    events_file = append_live_macro_event(event_type="publish", payload=payload, out_file=out_path)
    payload["events_file"] = events_file
    print(json.dumps(payload, ensure_ascii=True, indent=2) if args.json else json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
