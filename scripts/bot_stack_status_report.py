import argparse
import csv
import json
import gzip
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python without zoneinfo
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
WATCHDOG_DIR = PROJECT_ROOT / "governance" / "watchdog"
OUTPUT_DIR = PROJECT_ROOT / "exports" / "bot_stack_status"
DEFAULT_TOP = 10
_ET_ZONE = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone.utc

DECISION_LOGS = {
    "schwab_conservative": PROJECT_ROOT / "decision_explanations" / "shadow_conservative_equities" / "latest_decisions.log",
    "schwab_aggressive": PROJECT_ROOT / "decision_explanations" / "shadow_aggressive_equities" / "latest_decisions.log",
    "coinbase_crypto": PROJECT_ROOT / "decision_explanations" / "shadow_crypto" / "latest_decisions.log",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _last_lines(path: Path, max_lines: int = 1500) -> List[str]:
    text = _read_text(path)
    if not text:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _classify_lane(row: Dict[str, Any]) -> str:
    role = str(row.get("bot_role") or "").strip().lower()
    bot_id = str(row.get("bot_id") or "").strip().lower()

    if role == "infrastructure_sub_bot":
        return "infrastructure"
    if role == "options_sub_bot":
        return "options"
    if role == "futures_sub_bot":
        return "futures"
    if any(tok in bot_id for tok in ("long_term", "dividend_quality_compounder", "dividend_yield_trap_avoidance")):
        return "long_term"
    if any(tok in bot_id for tok in ("intraday", "scalp", "open_close", "ultrafast", "day_trade", "daytrading")):
        return "day"
    if any(tok in bot_id for tok in ("swing", "position_1m_3m", "1w_3w", "2d_5d")):
        return "swing"
    return "equities"


def _registry_summary(registry: Dict[str, Any], top_n: int) -> Dict[str, Any]:
    sub_bots = registry.get("sub_bots", []) if isinstance(registry, dict) else []
    sub_bots = sub_bots if isinstance(sub_bots, list) else []

    total = len(sub_bots)
    active_rows = [b for b in sub_bots if bool(b.get("active"))]
    inactive_rows = [b for b in sub_bots if not bool(b.get("active"))]
    deleted_rows = [b for b in sub_bots if bool(b.get("deleted_from_rotation"))]

    role_active = Counter(str(b.get("bot_role", "unknown")) for b in active_rows)
    role_total = Counter(str(b.get("bot_role", "unknown")) for b in sub_bots)

    active_sorted = sorted(
        active_rows,
        key=lambda b: (
            _safe_float(b.get("weight"), 0.0),
            _safe_float(b.get("quality_score"), 0.0),
            _safe_float(b.get("test_accuracy"), 0.0),
        ),
        reverse=True,
    )

    lane_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in active_sorted:
        lane_map[_classify_lane(row)].append(row)

    lane_summary: Dict[str, Any] = {}
    for lane, rows in sorted(
        lane_map.items(),
        key=lambda item: (
            -sum(_safe_float(r.get("weight"), 0.0) for r in item[1]),
            item[0],
        ),
    ):
        lane_summary[lane] = {
            "active_count": len(rows),
            "active_weight": round(sum(_safe_float(r.get("weight"), 0.0) for r in rows), 6),
            "roles": dict(Counter(str(r.get("bot_role") or "unknown") for r in rows)),
            "bots": [
                {
                    "bot_id": r.get("bot_id"),
                    "bot_role": r.get("bot_role"),
                    "weight": _safe_float(r.get("weight"), 0.0),
                    "quality_score": _safe_float(r.get("quality_score"), 0.0),
                    "test_accuracy": r.get("test_accuracy"),
                    "reason": r.get("reason"),
                }
                for r in rows[:top_n]
            ],
        }

    risk_rows = [
        b for b in sub_bots
        if not bool(b.get("deleted_from_rotation")) and _safe_int(b.get("no_improvement_streak"), 0) > 0
    ]
    risk_rows.sort(
        key=lambda b: (
            _safe_int(b.get("no_improvement_streak"), 0),
            _safe_float(b.get("quality_score"), 0.0) * -1.0,
        ),
        reverse=True,
    )

    return {
        "updated_at_utc": registry.get("updated_at_utc"),
        "policy": registry.get("master_policy", {}),
        "counts": {
            "total": total,
            "active": len(active_rows),
            "inactive": len(inactive_rows),
            "deleted": len(deleted_rows),
        },
        "roles": {
            "active": dict(role_active),
            "total": dict(role_total),
        },
        "lanes": lane_summary,
        "top_active": [
            {
                "bot_id": b.get("bot_id"),
                "bot_role": b.get("bot_role"),
                "weight": _safe_float(b.get("weight"), 0.0),
                "quality_score": _safe_float(b.get("quality_score"), 0.0),
                "test_accuracy": b.get("test_accuracy"),
                "reason": b.get("reason"),
            }
            for b in active_sorted[:top_n]
        ],
        "top_risk_streak": [
            {
                "bot_id": b.get("bot_id"),
                "bot_role": b.get("bot_role"),
                "streak": _safe_int(b.get("no_improvement_streak"), 0),
                "active": bool(b.get("active")),
                "deleted": bool(b.get("deleted_from_rotation")),
                "reason": b.get("reason"),
                "quality_score": _safe_float(b.get("quality_score"), 0.0),
            }
            for b in risk_rows[:top_n]
        ],
    }


def _line_action(line: str) -> Optional[str]:
    m = re.search(r"\baction=([A-Z_]+)\b", line)
    if not m:
        return None
    return m.group(1)


def _line_status(line: str) -> Optional[str]:
    m = re.search(r"\bstatus=([A-Z_]+)\b", line)
    if not m:
        return None
    return m.group(1)


def _line_symbol(line: str) -> Optional[str]:
    m = re.search(r"\bsymbol=([A-Za-z0-9._\-]+)\b", line)
    if not m:
        return None
    return m.group(1)


def _line_master_weights(line: str) -> Optional[Dict[str, float]]:
    m = re.search(r"master_weights=([^|]+)", line)
    if not m:
        return None
    raw = m.group(1).strip()
    out: Dict[str, float] = {}
    for pair in raw.split(","):
        if ":" not in pair:
            continue
        key, value = pair.split(":", 1)
        out[key.strip()] = _safe_float(value.strip(), 0.0)
    return out or None


def _line_top_contributors(line: str) -> List[str]:
    m = re.search(r"top_contributors=([^|]+)", line)
    if not m:
        return []
    raw = m.group(1).strip()
    items = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            items.append(token)
    return items


def _parse_decision_log(path: Path) -> Dict[str, Any]:
    lines = _last_lines(path, max_lines=8000)
    if not lines:
        return {
            "path": str(path),
            "exists": path.exists(),
            "lines_scanned": 0,
            "decision_lines": 0,
            "shadow_loops": 0,
            "actions": {},
            "statuses": {},
            "symbols_seen": 0,
            "latest_loop": None,
            "master_bias_counts": {},
            "grand_master": {},
            "options_master": {},
            "sub_bot_mentions": {},
            "top_contributors": {},
        }

    action_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    master_bias_counts: Counter[str] = Counter()
    sub_bot_mentions: Counter[str] = Counter()
    contributor_counts: Counter[str] = Counter()
    symbols = set()
    shadow_loops = 0
    latest_loop_line = None

    gm_scores: List[float] = []
    gm_weights: List[Dict[str, float]] = []
    options_scores: List[float] = []

    for line in lines:
        if "[ShadowLoop]" in line:
            shadow_loops += 1
            latest_loop_line = line
            continue
        if "[Decision]" not in line:
            continue

        action = _line_action(line)
        status = _line_status(line)
        symbol = _line_symbol(line)
        if action:
            action_counts[action] += 1
        if status:
            status_counts[status] += 1
        if symbol:
            symbols.add(symbol)

        bot_match = re.search(r"\bbot_id=([A-Za-z0-9_]+)\b", line)
        if bot_match:
            sub_bot_mentions[bot_match.group(1)] += 1

        if "trend_master_bias" in line:
            master_bias_counts["trend"] += 1
        if "mean_revert_master_bias" in line:
            master_bias_counts["mean_revert"] += 1
        if "shock_master_bias" in line:
            master_bias_counts["shock"] += 1
        if "grand_master_routing" in line:
            score_match = re.search(r"\bscore=([0-9.]+)\b", line)
            if score_match:
                gm_scores.append(_safe_float(score_match.group(1), 0.0))
            weights = _line_master_weights(line)
            if weights:
                gm_weights.append(weights)
        if "options_master_regime_filter" in line:
            score_match = re.search(r"\bscore=([0-9.]+)\b", line)
            if score_match:
                options_scores.append(_safe_float(score_match.group(1), 0.0))

        for token in _line_top_contributors(line):
            contributor_counts[token] += 1

    latest_loop: Optional[Dict[str, Any]] = None
    if latest_loop_line:
        iter_match = re.search(r"\biter=([0-9]+)\b", latest_loop_line)
        price_match = re.search(r"\bprice=([0-9.]+)\b", latest_loop_line)
        grand_action_match = re.search(r"\bgrand_action=([A-Z_]+)\b", latest_loop_line)
        options_action_match = re.search(r"\boptions_action=([A-Z_]+)\b", latest_loop_line)
        active_bots_match = re.search(r"\bactive_bots=([0-9]+)\b", latest_loop_line)
        recs_match = re.search(r"\brecs=([0-9]+)\b", latest_loop_line)
        snapshot_match = re.search(r"\bsnapshot_id=([A-Za-z0-9._\-]+)\b", latest_loop_line)
        latest_loop = {
            "raw": latest_loop_line,
            "iter": _safe_int(iter_match.group(1) if iter_match else None, 0),
            "symbol": _line_symbol(latest_loop_line),
            "price": _safe_float(price_match.group(1) if price_match else None, 0.0),
            "grand_action": grand_action_match.group(1) if grand_action_match else None,
            "options_action": options_action_match.group(1) if options_action_match else None,
            "active_bots": _safe_int(active_bots_match.group(1) if active_bots_match else None, 0),
            "recs": _safe_int(recs_match.group(1) if recs_match else None, 0),
            "snapshot_id": snapshot_match.group(1) if snapshot_match else None,
        }

    avg_gm_weights: Dict[str, float] = {}
    if gm_weights:
        agg: defaultdict[str, float] = defaultdict(float)
        for row in gm_weights:
            for k, v in row.items():
                agg[k] += v
        avg_gm_weights = {k: round(v / len(gm_weights), 4) for k, v in agg.items()}

    return {
        "path": str(path),
        "exists": path.exists(),
        "lines_scanned": len(lines),
        "decision_lines": sum(1 for line in lines if "[Decision]" in line),
        "shadow_loops": shadow_loops,
        "actions": dict(action_counts),
        "statuses": dict(status_counts),
        "symbols_seen": len(symbols),
        "latest_loop": latest_loop,
        "master_bias_counts": dict(master_bias_counts),
        "grand_master": {
            "samples": len(gm_scores),
            "avg_score": round(sum(gm_scores) / len(gm_scores), 6) if gm_scores else None,
            "avg_weights": avg_gm_weights,
        },
        "options_master": {
            "samples": len(options_scores),
            "avg_score": round(sum(options_scores) / len(options_scores), 6) if options_scores else None,
        },
        "sub_bot_mentions": dict(sub_bot_mentions.most_common(15)),
        "top_contributors": dict(contributor_counts.most_common(15)),
    }


def _resolve_watchdog_file() -> Optional[Path]:
    if not WATCHDOG_DIR.exists():
        return None
    candidates = sorted(
        list(WATCHDOG_DIR.glob('watchdog_events_*.jsonl')) + list(WATCHDOG_DIR.glob('watchdog_events_*.jsonl.gz')),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _parse_watchdog() -> Dict[str, Any]:
    source = _resolve_watchdog_file()
    if source is None:
        return {"exists": False, "path": str(WATCHDOG_DIR), "latest": None, "latest_timestamp_utc": None, "targets": []}

    latest: Optional[Dict[str, Any]] = None
    try:
        if source.suffix == '.gz':
            with gzip.open(source, 'rt', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        latest = json.loads(line)
                    except Exception:
                        continue
        else:
            with source.open('r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        latest = json.loads(line)
                    except Exception:
                        continue
    except Exception:
        pass

    if not latest:
        return {"exists": True, "path": str(source), "latest": None, "latest_timestamp_utc": None, "targets": []}

    return {
        "exists": True,
        "path": str(source),
        "latest_timestamp_utc": latest.get("timestamp_utc"),
        "targets": latest.get("targets", []) or [],
    }


def _session_window_et() -> tuple[int, int]:
    try:
        start_hour = int(float(os.getenv("MARKET_SESSION_START_HOUR", "8")))
    except Exception:
        start_hour = 8
    try:
        end_hour = int(float(os.getenv("MARKET_SESSION_END_HOUR", "20")))
    except Exception:
        end_hour = 20
    return max(start_hour, 0), min(max(end_hour, 0), 24)


def _flow_expected_live(name: str, now: Optional[datetime] = None) -> bool:
    local_now = now or datetime.now(timezone.utc).astimezone(_ET_ZONE)
    if str(name) == "coinbase_crypto":
        return True
    if not str(name).startswith("schwab_"):
        return True
    if local_now.weekday() >= 5:
        return False
    start_hour, end_hour = _session_window_et()
    current_hour = local_now.hour + (local_now.minute / 60.0)
    return float(start_hour) <= current_hour < float(end_hour)


def _overall_health(
    registry_counts: Dict[str, int],
    decision_summaries: Dict[str, Dict[str, Any]],
    watchdog: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    checks: List[Tuple[str, bool, str]] = []
    local_now = now or datetime.now(timezone.utc).astimezone(_ET_ZONE)

    active = _safe_int(registry_counts.get("active"), 0)
    checks.append(("active_sub_bots", active >= 15, f"active={active} (target>=15)"))

    loops_ok = True
    loop_notes: List[str] = []
    for name, summary in decision_summaries.items():
        has_loop = bool(summary.get("latest_loop"))
        has_decision_activity = _safe_int(summary.get("decision_lines"), 0) > 0
        expected_live = _flow_expected_live(name, local_now)
        healthy_activity = (has_loop or has_decision_activity) if expected_live else True
        loops_ok = loops_ok and healthy_activity
        state = "ok" if healthy_activity else "missing"
        if not expected_live:
            state = "off_session"
        loop_notes.append(
            f"{name}={state}(loop={'yes' if has_loop else 'no'},decisions={summary.get('decision_lines', 0)})"
        )
    checks.append(("live_shadow_loops", loops_ok, ", ".join(loop_notes)))

    targets = watchdog.get("targets", []) or []
    target_map = {str(t.get("name", "")): t for t in targets if isinstance(t, dict)}
    schwab_expected = any(_flow_expected_live(name, local_now) for name in decision_summaries if name.startswith("schwab_"))
    schwab_live = bool(target_map.get("schwab_parallel", {}).get("live")) or any(
        bool((decision_summaries.get(name) or {}).get("latest_loop")) or _safe_int((decision_summaries.get(name) or {}).get("decision_lines"), 0) > 0
        for name in decision_summaries
        if name.startswith("schwab_")
    )
    if not schwab_expected:
        schwab_live = True
        schwab_note = "off_session"
    else:
        schwab_note = f"live={bool(target_map.get('schwab_parallel', {}).get('live'))} fallback_activity={schwab_live}"
    coinbase_summary = decision_summaries.get("coinbase_crypto", {})
    coinbase_activity = bool(coinbase_summary.get("latest_loop")) or _safe_int(coinbase_summary.get("decision_lines"), 0) > 0
    coinbase_watchdog_live = bool(target_map.get("coinbase_shadow", {}).get("live"))
    coinbase_live = bool(coinbase_watchdog_live or coinbase_activity)
    coinbase_note = (
        f"live={coinbase_watchdog_live} fallback_activity={coinbase_activity}"
        if targets
        else f"watchdog_unavailable fallback_activity={coinbase_activity}"
    )
    checks.append(("watchdog_schwab_live", schwab_live, schwab_note))
    checks.append(("watchdog_coinbase_live", coinbase_live, coinbase_note))

    status = "healthy" if all(ok for _, ok, _ in checks) else "degraded"
    return {
        "status": status,
        "checks": [{"name": name, "ok": ok, "note": note} for name, ok, note in checks],
    }


def _flatten_for_numbers(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    rows.append(("generated_utc", str(payload.get("generated_utc"))))
    rows.append(("overall_status", str((payload.get("overall_health") or {}).get("status"))))

    counts = ((payload.get("registry") or {}).get("counts") or {})
    rows.append(("registry_total", str(counts.get("total", 0))))
    rows.append(("registry_active", str(counts.get("active", 0))))
    rows.append(("registry_inactive", str(counts.get("inactive", 0))))
    rows.append(("registry_deleted", str(counts.get("deleted", 0))))

    for chk in ((payload.get("overall_health") or {}).get("checks") or []):
        name = str((chk or {}).get("name", "check"))
        rows.append((f"check_{name}", str(bool((chk or {}).get("ok"))).lower()))

    for flow_name, summary in (payload.get("decision_logs") or {}).items():
        rows.append((f"{flow_name}_decision_lines", str(summary.get("decision_lines", 0))))
        rows.append((f"{flow_name}_symbols_seen", str(summary.get("symbols_seen", 0))))
        gm = summary.get("grand_master") or {}
        rows.append((f"{flow_name}_grand_master_avg_score", str(gm.get("avg_score"))))
        opt = summary.get("options_master") or {}
        rows.append((f"{flow_name}_options_master_avg_score", str(opt.get("avg_score"))))

    targets = (payload.get("watchdog") or {}).get("targets") or []
    for t in targets:
        name = str((t or {}).get("name", "target"))
        rows.append((f"watchdog_{name}_live", str(bool((t or {}).get("live"))).lower()))
        rows.append((f"watchdog_{name}_process_live", str(bool((t or {}).get("process_live"))).lower()))
        rows.append((f"watchdog_{name}_action", str((t or {}).get("action"))))

    return rows


def _write_numbers_csv(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in rows:
            w.writerow([k, v])


def _render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Bot Stack Status Report")
    lines.append("")
    lines.append(f"- generated_utc: `{payload['generated_utc']}`")
    lines.append(f"- overall_status: `{payload['overall_health']['status']}`")
    lines.append("")
    lines.append("## At A Glance")
    for check in payload["overall_health"]["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        lines.append(f"- {check['name']}: **{mark}** ({check['note']})")
    lines.append("")

    registry = payload["registry"]
    counts = registry["counts"]
    lines.append("## Sub-Bot Registry")
    lines.append(
        f"- total={counts['total']} active={counts['active']} inactive={counts['inactive']} deleted={counts['deleted']}"
    )
    if registry["roles"]["active"]:
        lines.append("- active_roles=" + ", ".join(f"{k}:{v}" for k, v in sorted(registry["roles"]["active"].items())))
    else:
        lines.append("- active_roles=none")
    lines.append("")
    lines.append("### Active By Lane")
    if registry["lanes"]:
        for lane, info in registry["lanes"].items():
            top = ", ".join(
                f"{row['bot_id']}({row['weight']:.4f})"
                for row in info.get("bots", [])[:4]
            ) or "none"
            lines.append(
                f"- {lane}: count={info['active_count']} weight={info['active_weight']:.4f} "
                f"roles={info['roles']} top={top}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("### Top Active (by weight)")
    if registry["top_active"]:
        for row in registry["top_active"]:
            lines.append(
                f"- {row['bot_id']} role={row['bot_role']} weight={row['weight']:.4f} "
                f"quality={row['quality_score']:.4f} acc={row['test_accuracy']}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("### Sleeve View")
    if registry["lanes"]:
        for lane, info in registry["lanes"].items():
            lines.append(f"- {lane}")
            for row in info.get("bots", []):
                lines.append(
                    f"  - {row['bot_id']} role={row['bot_role']} weight={row['weight']:.4f} "
                    f"quality={row['quality_score']:.4f} reason={row['reason']}"
                )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("### At-Risk Streak Bots")
    if registry["top_risk_streak"]:
        for row in registry["top_risk_streak"]:
            lines.append(
                f"- {row['bot_id']} streak={row['streak']} active={row['active']} deleted={row['deleted']} "
                f"quality={row['quality_score']:.4f} reason={row['reason']}"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Masterbot + Grand Master Flow")
    for name, summary in payload["decision_logs"].items():
        lines.append(f"### {name}")
        lines.append(
            f"- decisions={summary['decision_lines']} shadow_loops={summary['shadow_loops']} symbols_seen={summary['symbols_seen']}"
        )
        actions = ", ".join(f"{k}:{v}" for k, v in sorted(summary["actions"].items())) or "none"
        statuses = ", ".join(f"{k}:{v}" for k, v in sorted(summary["statuses"].items())) or "none"
        biases = ", ".join(f"{k}:{v}" for k, v in sorted(summary["master_bias_counts"].items())) or "none"
        lines.append(f"- actions={actions}")
        lines.append(f"- statuses={statuses}")
        lines.append(f"- master_bias_votes={biases}")
        gm = summary.get("grand_master") or {}
        opt = summary.get("options_master") or {}
        lines.append(
            f"- grand_master samples={gm.get('samples')} avg_score={gm.get('avg_score')} avg_weights={gm.get('avg_weights') or 'n/a'}"
        )
        lines.append(f"- options_master samples={opt.get('samples')} avg_score={opt.get('avg_score')}")
        latest = summary.get("latest_loop")
        if latest:
            lines.append(
                f"- latest_loop iter={latest.get('iter')} symbol={latest.get('symbol')} grand_action={latest.get('grand_action')} "
                f"options_action={latest.get('options_action')} active_bots={latest.get('active_bots')} recs={latest.get('recs')}"
            )
        else:
            lines.append("- latest_loop none")
        lines.append("- top_sub_bot_mentions=" + (", ".join(f"{k}:{v}" for k, v in (summary.get("sub_bot_mentions") or {}).items()) or "none"))
        lines.append("- top_contributors=" + (", ".join(f"{k}:{v}" for k, v in (summary.get("top_contributors") or {}).items()) or "none"))
        lines.append("")

    lines.append("## Watchdog")
    watchdog = payload["watchdog"]
    lines.append(f"- source={watchdog['path']}")
    lines.append(f"- latest_timestamp_utc={watchdog.get('latest_timestamp_utc')}")
    if watchdog["targets"]:
        for row in watchdog["targets"]:
            lines.append(
                f"- {row.get('name')}: live={row.get('live')} process_live={row.get('process_live')} "
                f"matches={row.get('match_count')} action={row.get('action')} note={row.get('note')}"
            )
    else:
        lines.append("- no_targets_found")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Readable bot + masterbot status report")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="Top N bots/contributors to include")
    parser.add_argument("--print", action="store_true", help="Print markdown report to stdout")
    args = parser.parse_args()

    top_n = max(_safe_int(args.top, DEFAULT_TOP), 1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    registry_raw = _load_registry(REGISTRY_PATH)
    registry = _registry_summary(registry_raw, top_n=top_n)

    decision_summaries = {
        name: _parse_decision_log(path)
        for name, path in DECISION_LOGS.items()
    }
    watchdog = _parse_watchdog()
    overall = _overall_health(registry["counts"], decision_summaries, watchdog)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "registry": registry,
        "decision_logs": decision_summaries,
        "watchdog": watchdog,
        "overall_health": overall,
        "sources": {
            "registry": str(REGISTRY_PATH),
            "watchdog_events": watchdog.get("path"),
            "decision_logs": {k: str(v) for k, v in DECISION_LOGS.items()},
        },
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = OUTPUT_DIR / f"bot_stack_status_{ts}.md"
    json_path = OUTPUT_DIR / f"bot_stack_status_{ts}.json"
    latest_md = OUTPUT_DIR / "latest.md"
    latest_json = OUTPUT_DIR / "latest.json"

    md_text = _render_markdown(payload)
    json_text = json.dumps(payload, ensure_ascii=True, indent=2)
    numbers_rows = _flatten_for_numbers(payload)

    numbers_csv_path = OUTPUT_DIR / f"bot_stack_status_{ts}.csv"
    latest_numbers_csv = OUTPUT_DIR / "latest.csv"

    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    _write_numbers_csv(numbers_csv_path, numbers_rows)
    _write_numbers_csv(latest_numbers_csv, numbers_rows)

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {numbers_csv_path}")
    print(f"Latest MD: {latest_md}")
    print(f"Latest JSON: {latest_json}")
    print(f"Latest CSV: {latest_numbers_csv}")

    if args.print:
        print("")
        print(md_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
