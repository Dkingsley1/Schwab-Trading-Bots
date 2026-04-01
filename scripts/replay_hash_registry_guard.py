import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALTH = PROJECT_ROOT / "governance" / "health"
ALERTS = PROJECT_ROOT / "governance" / "alerts"


def _load(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _key(profile: str, domain: str) -> str:
    p = str(profile or "all").strip().lower() or "all"
    d = str(domain or "all").strip().lower() or "all"
    return f"{p}|{d}"


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _should_auto_rebaseline(
    node: dict[str, Any],
    *,
    now: datetime,
    stale_days: float,
    source_ok: bool,
    source_hash_match: bool = False,
) -> bool:
    if not source_ok:
        return False
    # If the source replay artifact already declares its current hash healthy,
    # prefer syncing the registry immediately instead of waiting for age-based
    # staleness to expire.
    if bool(source_hash_match):
        return True
    updated = _parse_iso_utc(node.get("updated_utc"))
    if updated is None:
        return True
    age_days = max((now - updated).total_seconds(), 0.0) / 86400.0
    return age_days >= max(float(stale_days), 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Persist expected replay hashes and alert on drift by profile/domain.")
    ap.add_argument("--registry-file", default=str(HEALTH / "replay_expected_hashes.json"))
    ap.add_argument("--paper-file", default=str(HEALTH / "paper_replay_drill_latest.json"))
    ap.add_argument("--e2e-file", default=str(HEALTH / "replay_end_to_end_latest.json"))
    ap.add_argument("--bootstrap-if-missing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--auto-rebaseline-stale-days", type=float, default=14.0)
    ap.add_argument("--alert-file", default=str(ALERTS / "replay_hash_drift_events.jsonl"))
    ap.add_argument("--latest-alert-file", default=str(ALERTS / "replay_hash_drift_latest.json"))
    ap.add_argument("--out-file", default=str(HEALTH / "replay_hash_registry_guard_latest.json"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    registry_path = Path(args.registry_file)
    reg = _load(registry_path)
    reg.setdefault("paper_replay", {})
    reg.setdefault("replay_end_to_end", {})

    paper = _load(Path(args.paper_file))
    e2e = _load(Path(args.e2e_file))

    failed = []
    details: dict[str, Any] = {"paper": {}, "e2e": {}}

    paper_hash = str(paper.get("replay_hash", "")).strip().lower()
    paper_profile = str((paper.get("profile") or "all")).strip().lower() or "all"
    paper_domain = str((paper.get("domain") or "all")).strip().lower() or "all"
    pkey = _key(paper_profile, paper_domain)
    pnode = reg["paper_replay"].get(pkey, {}) if isinstance(reg.get("paper_replay"), dict) else {}
    pexp = str((pnode or {}).get("expected_hash", "")).strip().lower()

    if paper_hash:
        if not pexp and args.bootstrap_if_missing:
            reg["paper_replay"][pkey] = {"expected_hash": paper_hash, "updated_utc": now}
            pexp = paper_hash
        elif pexp and paper_hash != pexp and _should_auto_rebaseline(
            pnode if isinstance(pnode, dict) else {},
            now=now_dt,
            stale_days=float(args.auto_rebaseline_stale_days),
            source_ok=bool(paper.get("ok", False)),
            source_hash_match=bool(paper.get("hash_match", False)),
        ):
            reg["paper_replay"][pkey] = {"expected_hash": paper_hash, "updated_utc": now, "auto_rebased": True}
            pexp = paper_hash
        pmatch = bool(pexp and paper_hash == pexp)
        details["paper"] = {
            "profile": paper_profile,
            "domain": paper_domain,
            "key": pkey,
            "expected_hash": pexp,
            "current_hash": paper_hash,
            "hash_match": pmatch,
            "auto_rebased": bool(isinstance(reg["paper_replay"].get(pkey), dict) and reg["paper_replay"][pkey].get("auto_rebased")),
        }
        if pexp and (not pmatch):
            failed.append("paper_replay_hash_drift")

    e2e_hash = str(e2e.get("replay_hash", "")).strip().lower()
    ekey = "global"
    enode = reg["replay_end_to_end"].get(ekey, {}) if isinstance(reg.get("replay_end_to_end"), dict) else {}
    eexp = str((enode or {}).get("expected_hash", "")).strip().lower()
    if e2e_hash:
        if not eexp and args.bootstrap_if_missing:
            reg["replay_end_to_end"][ekey] = {"expected_hash": e2e_hash, "updated_utc": now}
            eexp = e2e_hash
        elif eexp and e2e_hash != eexp and _should_auto_rebaseline(
            enode if isinstance(enode, dict) else {},
            now=now_dt,
            stale_days=float(args.auto_rebaseline_stale_days),
            source_ok=bool(e2e.get("ok", False)),
            source_hash_match=bool(e2e.get("hash_match", False)),
        ):
            reg["replay_end_to_end"][ekey] = {"expected_hash": e2e_hash, "updated_utc": now, "auto_rebased": True}
            eexp = e2e_hash
        ematch = bool(eexp and e2e_hash == eexp)
        details["e2e"] = {
            "key": ekey,
            "expected_hash": eexp,
            "current_hash": e2e_hash,
            "hash_match": ematch,
            "auto_rebased": bool(isinstance(reg["replay_end_to_end"].get(ekey), dict) and reg["replay_end_to_end"][ekey].get("auto_rebased")),
        }
        if eexp and (not ematch):
            failed.append("e2e_replay_hash_drift")

    out = {
        "timestamp_utc": now,
        "ok": len(failed) == 0,
        "failed_checks": failed,
        "details": details,
        "registry_file": str(registry_path),
    }

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(reg, ensure_ascii=True, indent=2), encoding="utf-8")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if failed:
        alert = {"timestamp_utc": now, "event": "replay_hash_drift", "failed_checks": failed, "details": details}
        _append_jsonl(Path(args.alert_file), alert)
        latest = Path(args.latest_alert_file)
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps(alert, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(f"replay_hash_registry_guard_ok={int(out['ok'])} failed_checks={','.join(failed) if failed else 'none'}")
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
