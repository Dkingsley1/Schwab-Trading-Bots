#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs" / "showcase"
GENERATED_ROOT = DOCS_ROOT / "generated"
README_PATH = PROJECT_ROOT / "README.md"
HIGHLIGHTS_JSON = GENERATED_ROOT / "highlights_latest.json"
HIGHLIGHTS_MD = GENERATED_ROOT / "highlights_latest.md"
README_START = "<!-- SHOWCASE_HIGHLIGHTS_START -->"
README_END = "<!-- SHOWCASE_HIGHLIGHTS_END -->"


def _safe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _fmt_pct(raw: float | None) -> str:
    if raw is None:
        return "n/a"
    return f"{raw * 100.0:.1f}%"


def _fmt_ratio_pct(raw: float | None) -> str:
    if raw is None:
        return "n/a"
    return f"{raw * 100.0:.2f}%"


def _fmt_compact_timestamp(raw: Any) -> str:
    if not raw:
        return "unknown"
    text = str(raw).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _active_bot_summary() -> dict[str, Any]:
    registry = _safe_load_json(PROJECT_ROOT / "master_bot_registry.json", default={})
    sub_bots = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    active = [row for row in sub_bots if isinstance(row, Mapping) and bool(row.get("active"))]
    roles = Counter(str(row.get("bot_role") or "unknown") for row in active)
    protected_lanes = registry.get("master_policy", {}).get("protected_collection_lane_floors", {}) if isinstance(registry.get("master_policy"), Mapping) else {}
    top_active = []
    for row in sorted(active, key=lambda item: _safe_float(item.get("test_accuracy"), -1.0), reverse=True)[:5]:
        top_active.append(
            {
                "bot_id": row.get("bot_id"),
                "bot_role": row.get("bot_role"),
                "test_accuracy": _safe_float(row.get("test_accuracy"), 0.0),
                "quality_score": _safe_float(row.get("quality_score"), 0.0),
                "reason": row.get("reason"),
            }
        )
    return {
        "total_registered": len(sub_bots),
        "active_count": len(active),
        "active_roles": dict(roles),
        "protected_collection_lane_floors": protected_lanes,
        "top_active_bots": top_active,
    }


def _live_lane_summary() -> dict[str, Any]:
    health_dir = PROJECT_ROOT / "governance" / "health"
    lane_files = sorted(health_dir.glob("data_ingress_latest_*.json"))
    lanes = []
    running = 0
    for path in lane_files:
        payload = _safe_load_json(path, default={})
        if not isinstance(payload, Mapping):
            continue
        loop_state = str(payload.get("loop_state") or "").strip().lower()
        lane_name = path.stem.replace("data_ingress_latest_", "")
        lanes.append(
            {
                "lane": lane_name,
                "loop_state": loop_state,
                "iter": int(_safe_float(payload.get("iter"), 0.0)),
                "api_error": int(_safe_float(payload.get("api_error"), 0.0)),
            }
        )
        if loop_state == "running":
            running += 1
    return {
        "lane_count": len(lanes),
        "running_count": running,
        "lanes": lanes,
    }


def _artifact_snapshot() -> dict[str, Any]:
    health_dir = PROJECT_ROOT / "governance" / "health"
    reports_dir = PROJECT_ROOT / "exports" / "reports"
    crypto_ctx = _safe_load_json(health_dir / "crypto_market_context_sync_latest.json", default={})
    divergence = _safe_load_json(health_dir / "data_source_divergence_latest.json", default={})
    correlation = _safe_load_json(health_dir / "market_crypto_correlation_sync_latest.json", default={})
    training = _safe_load_json(health_dir / "training_success_latest.json", default={})
    watchdog = _safe_load_json(health_dir / "shadow_watchdog_tripwire_latest.json", default={})
    daily_ops = _safe_load_json(reports_dir / "daily_ops_report_latest.json", default={})
    return {
        "crypto_context": crypto_ctx if isinstance(crypto_ctx, Mapping) else {},
        "divergence": divergence if isinstance(divergence, Mapping) else {},
        "correlation": correlation if isinstance(correlation, Mapping) else {},
        "training": training if isinstance(training, Mapping) else {},
        "watchdog": watchdog if isinstance(watchdog, Mapping) else {},
        "daily_ops": daily_ops if isinstance(daily_ops, Mapping) else {},
    }


def _build_snapshot() -> dict[str, Any]:
    bot_summary = _active_bot_summary()
    lane_summary = _live_lane_summary()
    artifacts = _artifact_snapshot()
    crypto_ctx = artifacts["crypto_context"]
    divergence = artifacts["divergence"]
    correlation = artifacts["correlation"]
    training = artifacts["training"]
    watchdog = artifacts["watchdog"]
    daily_ops = artifacts["daily_ops"]

    highlights = [
        (
            f"Registry currently tracks {bot_summary['total_registered']} bots with {bot_summary['active_count']} active "
            f"across {', '.join(sorted(bot_summary['active_roles'])) or 'no active roles'} lanes."
        ),
        (
            f"Live ingestion is wired across {lane_summary['lane_count']} lane artifacts with "
            f"{lane_summary['running_count']} currently reporting `running`."
        ),
        (
            f"Crypto context is aggregating {int(_safe_float(crypto_ctx.get('ok_source_count'), 0.0))}/"
            f"{int(_safe_float(crypto_ctx.get('source_count'), 0.0))} healthy sources and "
            f"{int(_safe_float(crypto_ctx.get('news_ok_source_count'), 0.0))}/"
            f"{int(_safe_float(crypto_ctx.get('news_source_count'), 0.0))} healthy crypto news feeds."
        ),
        (
            f"Latest divergence check is `ok={bool(divergence.get('ok', False))}` with worst relative spread "
            f"{_fmt_ratio_pct(_safe_float(divergence.get('worst_relative_spread'), 0.0))}."
        ),
        (
            f"Market/crypto correlation overlay is running in `{str(correlation.get('mode') or 'exact')}` mode with "
            f"{int(_safe_float(correlation.get('aligned_pairs'), 0.0))} aligned pairs "
            f"and cache hits/misses {int(_safe_float(correlation.get('cache_hits'), 0.0))}/"
            f"{int(_safe_float(correlation.get('cache_misses'), 0.0))}."
        ),
        (
            f"Latest training summary: {int(_safe_float(training.get('trained_count'), 0.0))} trained, "
            f"{int(_safe_float(training.get('failure_count'), 0.0))} failed, "
            f"`confirmed_training_success={bool(training.get('confirmed_training_success', False))}`."
        ),
        (
            f"Watchdog tripwire is `active={bool(watchdog.get('active', False))}` and the latest daily ops "
            f"quality score is {daily_ops.get('quality', {}).get('data_quality_score', 'n/a')}."
        ),
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bot_summary": bot_summary,
        "lane_summary": lane_summary,
        "artifacts": artifacts,
        "highlights": highlights,
    }


def _render_highlights_markdown(snapshot: Mapping[str, Any]) -> str:
    bot_summary = snapshot["bot_summary"]
    lane_summary = snapshot["lane_summary"]
    artifacts = snapshot["artifacts"]
    training = artifacts["training"]
    correlation = artifacts["correlation"]
    crypto_ctx = artifacts["crypto_context"]
    top_bots = bot_summary["top_active_bots"]

    lines = [
        "# Auto-Refreshed Highlights",
        "",
        f"_Generated at {_fmt_compact_timestamp(snapshot.get('generated_at_utc'))}_",
        "",
        "## Platform Snapshot",
        "",
        f"- Registered bots: `{bot_summary['total_registered']}`",
        f"- Active bots: `{bot_summary['active_count']}`",
        f"- Live lane artifacts tracked: `{lane_summary['lane_count']}`",
        f"- Running lane artifacts: `{lane_summary['running_count']}`",
        f"- Crypto source coverage: `{int(_safe_float(crypto_ctx.get('ok_source_count'), 0.0))}/{int(_safe_float(crypto_ctx.get('source_count'), 0.0))}`",
        f"- Crypto news coverage: `{int(_safe_float(crypto_ctx.get('news_ok_source_count'), 0.0))}/{int(_safe_float(crypto_ctx.get('news_source_count'), 0.0))}`",
        f"- Correlation mode: `{str(correlation.get('mode') or 'exact')}`",
        f"- Last training result: `{int(_safe_float(training.get('trained_count'), 0.0))} trained / {int(_safe_float(training.get('failure_count'), 0.0))} failed`",
        "",
        "## Key Highlights",
        "",
    ]
    lines.extend(f"- {row}" for row in snapshot["highlights"])
    lines.extend(["", "## Current Active Lineup", ""])
    if top_bots:
        lines.extend(
            [
                "| Bot | Role | Test Accuracy | Quality Score |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in top_bots:
            lines.append(
                "| {bot} | {role} | {acc} | {quality:.3f} |".format(
                    bot=row.get("bot_id"),
                    role=row.get("bot_role"),
                    acc=_fmt_pct(_safe_float(row.get("test_accuracy"), 0.0)),
                    quality=_safe_float(row.get("quality_score"), 0.0),
                )
            )
    else:
        lines.append("- No active bots were found in the registry snapshot.")

    lines.extend(
        [
            "",
            "## Showcase Links",
            "",
            "- [Showcase Index](../README.md)",
            "- [Live Multi-Asset Paper Trading Platform](../projects/01-live-multi-asset-paper-platform.md)",
            "- [Quant Research and Model Training System](../projects/02-quant-research-and-model-training.md)",
            "- [Data Fusion and Verification Pipeline](../projects/03-data-fusion-and-verification-pipeline.md)",
            "- [Reliability, Safety, and Ops Automation](../projects/04-reliability-safety-and-ops-automation.md)",
            "- [Cross-Market Crypto and Macro Intelligence](../projects/05-cross-market-crypto-and-macro-intelligence.md)",
            "",
        ]
    )
    return "\n".join(lines)


def _render_readme_snippet(snapshot: Mapping[str, Any]) -> str:
    bot_summary = snapshot["bot_summary"]
    lane_summary = snapshot["lane_summary"]
    artifacts = snapshot["artifacts"]
    correlation = artifacts["correlation"]
    crypto_ctx = artifacts["crypto_context"]
    top_bots = bot_summary["top_active_bots"][:3]
    lines = [
        f"_Generated at {_fmt_compact_timestamp(snapshot.get('generated_at_utc'))}_",
        "",
        f"- Active registry lineup: `{bot_summary['active_count']}` of `{bot_summary['total_registered']}` bots are active.",
        f"- Live collection snapshot: `{lane_summary['running_count']}/{lane_summary['lane_count']}` lane artifacts are reporting `running`.",
        f"- Crypto context: `{int(_safe_float(crypto_ctx.get('ok_source_count'), 0.0))}/{int(_safe_float(crypto_ctx.get('source_count'), 0.0))}` healthy sources and `{int(_safe_float(crypto_ctx.get('news_ok_source_count'), 0.0))}/{int(_safe_float(crypto_ctx.get('news_source_count'), 0.0))}` healthy news feeds.",
        f"- Correlation overlay: mode `{str(correlation.get('mode') or 'exact')}`, aligned pairs `{int(_safe_float(correlation.get('aligned_pairs'), 0.0))}`.",
    ]
    if top_bots:
        formatted = ", ".join(f"`{row['bot_id']}` ({_fmt_pct(_safe_float(row['test_accuracy'], 0.0))})" for row in top_bots)
        lines.append(f"- Top active lineup by test accuracy: {formatted}.")
    lines.append("")
    lines.append("Full generated detail lives in [docs/showcase/generated/highlights_latest.md](docs/showcase/generated/highlights_latest.md).")
    return "\n".join(lines)


def _update_readme(snippet: str) -> None:
    text = README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(README_START)}.*?{re.escape(README_END)}",
        flags=re.DOTALL,
    )
    replacement = f"{README_START}\n{snippet}\n{README_END}"
    if pattern.search(text):
        text = pattern.sub(replacement, text)
    else:
        text = text.rstrip() + "\n\n## Auto-Refreshed Highlights\n\n" + replacement + "\n"
    README_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    snapshot = _build_snapshot()
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    HIGHLIGHTS_JSON.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    HIGHLIGHTS_MD.write_text(_render_highlights_markdown(snapshot) + "\n", encoding="utf-8")
    _update_readme(_render_readme_snippet(snapshot))
    print(json.dumps({"ok": True, "generated_at_utc": snapshot["generated_at_utc"], "output": str(HIGHLIGHTS_MD)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
