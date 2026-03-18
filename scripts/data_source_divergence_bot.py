import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALTH_DIR = PROJECT_ROOT / "governance" / "health"
GLOBAL_OUT = HEALTH_DIR / "data_source_divergence_latest.json"
BOND_OUT = HEALTH_DIR / "data_source_divergence_bond_latest.json"
NON_BOND_OUT = HEALTH_DIR / "data_source_divergence_non_bond_latest.json"


def _parse_ts(raw):
    if not raw:
        return None
    s = str(raw).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def _scope_for_shadow_dir(shadow_dir: str) -> str:
    token = str(shadow_dir or "").strip().lower()
    if token.startswith("shadow_bond_"):
        return "bond_profile"
    if token.startswith("shadow_"):
        return "non_bond_profiles"
    return "other_profiles"


def _meta_key(**kwargs):
    return tuple(sorted((str(k), str(v)) for k, v in kwargs.items() if str(v).strip()))


def _summarize_bucket_map(
    bucket_map,
    *,
    max_relative_spread: float,
    timestamp_utc: str,
    window_hours: int,
    comparison_mode: str,
    scope: str = "",
    profile_dirs=None,
):
    worst_rel = 0.0
    compared = 0
    offenders = []
    for meta_key, prices in bucket_map.items():
        if len(prices) < 2:
            continue
        meta = dict(meta_key)
        mn = min(prices)
        mx = max(prices)
        rel = (mx - mn) / max(mn, 1e-8)
        compared += 1
        if rel > worst_rel:
            worst_rel = rel
        if rel > float(max_relative_spread):
            row = {
                "symbol": meta.get("symbol", ""),
                "minute": meta.get("minute", ""),
                "rel_spread": round(rel, 6),
                "n": len(prices),
            }
            shadow_dir = str(meta.get("shadow_dir", "") or "")
            if shadow_dir:
                row["shadow_dir"] = shadow_dir
            offenders.append(row)

    payload = {
        "timestamp_utc": timestamp_utc,
        "ok": len(offenders) == 0,
        "window_hours": int(window_hours),
        "compared_buckets": int(compared),
        "worst_relative_spread": round(worst_rel, 6),
        "max_relative_spread": float(max_relative_spread),
        "offenders": offenders[:50],
        "comparison_mode": comparison_mode,
    }
    if scope:
        payload["scope"] = scope
    if profile_dirs is not None:
        payload["profile_dirs"] = sorted({str(x) for x in (profile_dirs or []) if str(x).strip()})
    return payload


def build_divergence_payloads(project_root: Path, *, hours: int, max_relative_spread: float):
    since = datetime.now(timezone.utc) - timedelta(hours=max(hours, 1))

    global_buckets = defaultdict(list)
    scope_buckets = defaultdict(lambda: defaultdict(list))
    profile_buckets = defaultdict(lambda: defaultdict(list))
    scope_dirs = defaultdict(set)

    for path in (project_root / "governance").glob("shadow*/master_control_*.jsonl"):
        shadow_dir = path.parent.name
        scope = _scope_for_shadow_dir(shadow_dir)
        scope_dirs[scope].add(shadow_dir)
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts = _parse_ts(row.get("timestamp_utc"))
                    if ts is None or ts < since:
                        continue
                    sym = str(row.get("symbol", "") or "").strip().upper()
                    if not sym:
                        continue
                    market = row.get("market", {}) if isinstance(row.get("market", {}), dict) else {}
                    px = float(market.get("last_price", 0.0) or 0.0)
                    if px <= 0:
                        continue
                    minute = ts.replace(second=0, microsecond=0).isoformat()
                    global_buckets[_meta_key(symbol=sym, minute=minute)].append(px)
                    scope_buckets[scope][_meta_key(shadow_dir=shadow_dir, symbol=sym, minute=minute)].append(px)
                    profile_buckets[shadow_dir][_meta_key(shadow_dir=shadow_dir, symbol=sym, minute=minute)].append(px)
        except Exception:
            continue

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    global_payload = _summarize_bucket_map(
        global_buckets,
        max_relative_spread=max_relative_spread,
        timestamp_utc=timestamp_utc,
        window_hours=hours,
        comparison_mode="cross_profile",
        scope="all_profiles",
        profile_dirs=sorted({d for dirs in scope_dirs.values() for d in dirs}),
    )

    scope_payloads = {}
    for scope, bucket_map in scope_buckets.items():
        scope_payloads[scope] = _summarize_bucket_map(
            bucket_map,
            max_relative_spread=max_relative_spread,
            timestamp_utc=timestamp_utc,
            window_hours=hours,
            comparison_mode="within_profile",
            scope=scope,
            profile_dirs=scope_dirs.get(scope, set()),
        )

    for scope in ("bond_profile", "non_bond_profiles"):
        scope_payloads.setdefault(
            scope,
            _summarize_bucket_map(
                {},
                max_relative_spread=max_relative_spread,
                timestamp_utc=timestamp_utc,
                window_hours=hours,
                comparison_mode="within_profile",
                scope=scope,
                profile_dirs=scope_dirs.get(scope, set()),
            ),
        )

    profile_payloads = {}
    for shadow_dir, bucket_map in profile_buckets.items():
        profile_payloads[shadow_dir] = _summarize_bucket_map(
            bucket_map,
            max_relative_spread=max_relative_spread,
            timestamp_utc=timestamp_utc,
            window_hours=hours,
            comparison_mode="within_profile",
            scope=shadow_dir,
            profile_dirs=[shadow_dir],
        )

    global_payload["scopes"] = scope_payloads
    global_payload["profiles"] = profile_payloads
    return global_payload, scope_payloads


def main() -> int:
    parser = argparse.ArgumentParser(description="Data source divergence bot.")
    parser.add_argument("--hours", type=int, default=2)
    parser.add_argument("--max-relative-spread", type=float, default=0.03)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload, scopes = build_divergence_payloads(
        PROJECT_ROOT,
        hours=int(args.hours),
        max_relative_spread=float(args.max_relative_spread),
    )

    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_OUT.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    BOND_OUT.write_text(json.dumps(scopes.get("bond_profile", {}), ensure_ascii=True, indent=2), encoding="utf-8")
    NON_BOND_OUT.write_text(json.dumps(scopes.get("non_bond_profiles", {}), ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print("data_source_divergence_ok=" + str(payload["ok"]).lower() + f" worst={payload['worst_relative_spread']}")

    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
