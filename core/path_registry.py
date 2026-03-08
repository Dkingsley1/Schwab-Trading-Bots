from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ShadowPathContext:
    profile: str
    domain: str
    broker: str

    @property
    def profile_tag(self) -> str:
        return self.profile if self.profile else "default"

    @property
    def key(self) -> str:
        return f"{self.profile_tag}_{self.domain}_{self.broker}"

    @property
    def shadow_subdir(self) -> str:
        base = "shadow" if not self.profile else f"shadow_{self.profile}"
        return f"{base}_{self.domain}" if self.domain else base


def _safe_token(raw: str) -> str:
    text = str(raw or "").strip().lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    value = "".join(out).strip("_")
    return value or "default"


def utc_day(now: Optional[datetime] = None) -> str:
    dt = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d")


def build_shadow_context(*, profile: str = "", domain: str = "", broker: str = "") -> ShadowPathContext:
    prof = _safe_token(profile or os.getenv("SHADOW_PROFILE", ""))
    if prof == "default":
        prof = ""

    dom_raw = (domain or os.getenv("SHADOW_DOMAIN", "")).strip().lower()
    brk = _safe_token((broker or os.getenv("DATA_BROKER", "schwab")).strip().lower() or "schwab")
    if dom_raw not in {"equities", "crypto"}:
        dom_raw = "crypto" if brk == "coinbase" else "equities"
    dom = _safe_token(dom_raw)
    return ShadowPathContext(profile=prof, domain=dom, broker=brk)


def governance_master_control_path(project_root: str | Path, ctx: ShadowPathContext, *, day: str = "") -> str:
    stamp = day or utc_day()
    return str(Path(project_root) / "governance" / ctx.shadow_subdir / f"master_control_{stamp}.jsonl")


def shadow_pnl_attribution_path(project_root: str | Path, ctx: ShadowPathContext, *, day: str = "") -> str:
    stamp = day or utc_day()
    return str(Path(project_root) / "governance" / ctx.shadow_subdir / f"shadow_pnl_attribution_{stamp}.jsonl")


def runtime_event_legacy_path(project_root: str | Path, *, day: str = "") -> str:
    stamp = day or utc_day()
    return str(Path(project_root) / "governance" / "events" / f"runtime_events_{stamp}.jsonl")


def api_calls_legacy_path(project_root: str | Path, ctx: ShadowPathContext, *, day: str = "") -> str:
    stamp = day or utc_day()
    return str(Path(project_root) / "governance" / "events" / f"api_calls_{ctx.key}_{stamp}.jsonl")


def loop_state_legacy_path(project_root: str | Path, ctx: ShadowPathContext, *, day: str = "") -> str:
    stamp = day or utc_day()
    return str(Path(project_root) / "governance" / "events" / f"loop_state_{ctx.key}_{stamp}.jsonl")


def gate_logs_legacy_path(project_root: str | Path, ctx: ShadowPathContext, *, day: str = "") -> str:
    stamp = day or utc_day()
    return str(Path(project_root) / "governance" / "events" / f"gate_logs_{ctx.key}_{stamp}.jsonl")


def data_ingress_legacy_path(project_root: str | Path, ctx: ShadowPathContext, *, day: str = "") -> str:
    stamp = day or utc_day()
    return str(Path(project_root) / "governance" / "events" / f"data_ingress_{ctx.key}_{stamp}.jsonl")


def channel_event_path(project_root: str | Path, channel: str, ctx: ShadowPathContext, *, day: str = "") -> str:
    stamp = day or utc_day()
    ch = _safe_token(channel)
    return str(Path(project_root) / "governance" / "channels" / ch / ctx.key / f"{ch}_{stamp}.jsonl")


def channel_snapshot_path(project_root: str | Path, channel: str, ctx: ShadowPathContext) -> str:
    ch = _safe_token(channel)
    return str(Path(project_root) / "governance" / "health" / "channels" / f"{ch}_latest_{ctx.key}.json")


def channel_cursor_path(project_root: str | Path, consumer: str, channel: str) -> str:
    c = _safe_token(consumer)
    ch = _safe_token(channel)
    return str(Path(project_root) / "governance" / "health" / "channel_cursors" / c / f"{ch}.json")


def decision_log_path(project_root: str | Path, subdir: str = "decisions", *, day: str = "") -> str:
    stamp = day or utc_day()
    return str(Path(project_root) / subdir / f"trade_decisions_{stamp}.jsonl")


def decision_explanations_paths(project_root: str | Path, mode_label: str, *, day: str = "") -> tuple[str, str]:
    stamp = day or utc_day()
    base = Path(project_root) / "decision_explanations" / _safe_token(mode_label)
    return str(base / f"decision_explanations_{stamp}.jsonl"), str(base / "latest_decisions.log")


def auth_events_path(project_root: str | Path, *, day: str = "") -> str:
    stamp = day or utc_day()
    return str(Path(project_root) / "governance" / "events" / f"auth_events_{stamp}.jsonl")


def execution_guard_path(project_root: str | Path, mode: str, *, day: str = "") -> str:
    stamp = day or utc_day()
    prefix = "paper_execution_guard" if str(mode).strip().lower() == "paper" else "live_execution_guard"
    return str(Path(project_root) / "governance" / "events" / f"{prefix}_{stamp}.jsonl")


def live_softguard_path(project_root: str | Path, *, day: str = "") -> str:
    stamp = day or utc_day()
    return str(Path(project_root) / "governance" / "events" / f"live_softguard_{stamp}.jsonl")


def ingress_state_path(project_root: str | Path, ctx: ShadowPathContext) -> str:
    return str(Path(project_root) / "governance" / "health" / f"data_ingress_latest_{ctx.key}.json")


def classify_channel_from_path(path: str) -> str:
    name = Path(path).name.lower()
    if name.startswith("runtime_events_") or name.startswith("runtime_"):
        return "runtime"
    if name.startswith("gate_logs_") or name.startswith("gate_"):
        return "gate"
    if name.startswith("data_ingress_") or name.startswith("ingress_"):
        return "ingress"
    if name.startswith("api_calls_"):
        return "api"
    if name.startswith("loop_state_"):
        return "loop_state"
    if name.startswith("master_control_") or name.startswith("trade_decisions_") or name.startswith("decision_explanations_"):
        return "decision"
    if name.startswith("shadow_pnl_attribution_") or "risk" in name:
        return "risk"
    if name.startswith("auth_events_"):
        return "auth"
    if name.startswith("paper_execution_guard_") or name.startswith("live_execution_guard_"):
        return "execution_guard"
    if name.startswith("live_softguard_"):
        return "softguard"
    return ""


def default_channel_mirror_paths(path: str, *, project_root: str | Path, ctx: Optional[ShadowPathContext] = None) -> list[str]:
    channel = classify_channel_from_path(path)
    if not channel:
        return []
    if str(path).startswith(str(Path(project_root) / "governance" / "channels")):
        return []

    context = ctx or build_shadow_context()
    mirror = channel_event_path(project_root, channel, context)
    if os.path.abspath(mirror) == os.path.abspath(path):
        return []
    return [mirror]
