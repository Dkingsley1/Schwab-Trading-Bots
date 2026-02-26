import json
import math
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional


TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{8}_\d{6}$")


@dataclass
class BotOutcome:
    bot_id: str
    log_file: str
    model_path: Optional[str]
    test_accuracy: Optional[float]
    best_val_f1: Optional[float]
    macro_f1: Optional[float]
    final_val_loss: Optional[float]
    max_drawdown: Optional[float]
    raw_metrics: Dict[str, float]


@dataclass
class BotStatus:
    bot_id: str
    bot_role: str
    active: bool
    reason: str
    weight: float
    preference_score: float
    quality_score: float
    test_accuracy: Optional[float]
    candidate_test_accuracy: Optional[float]
    candidate_quality_score: float
    previous_best_accuracy: Optional[float]
    no_improvement_streak: int
    deleted_from_rotation: bool
    delete_reason: str
    promoted: bool
    promotion_reason: str
    model_path: Optional[str]
    log_file: str
    candidate_log_file: str = ""


class MasterBot:
    """Orchestrates sub-bot lifecycle from training outcomes."""

    def __init__(
        self,
        project_root: str,
        preferred_low: float = 0.55,
        preferred_high: float = 0.65,
        deactivate_below: float = 0.50,
        quality_floor: float = 0.52,
        flash_crash_weight_cap: float = 0.35,
        decay_guard_drop: float = 0.08,
        promotion_margin: float = 0.005,
        no_improvement_retire_streak: int = 3,
        min_active_bots: int = 20,
        correlation_prune_threshold: float = 0.92,
    ) -> None:
        self.project_root = project_root
        self.logs_dir = os.path.join(project_root, "logs")
        self.registry_path = os.path.join(project_root, "master_bot_registry.json")
        self.preferred_low = preferred_low
        self.preferred_high = preferred_high
        self.deactivate_below = deactivate_below
        self.quality_floor = quality_floor
        self.flash_crash_weight_cap = flash_crash_weight_cap
        self.decay_guard_drop = decay_guard_drop
        self.promotion_margin = promotion_margin
        self.no_improvement_retire_streak = max(int(no_improvement_retire_streak), 1)
        self.max_active_no_improvement_streak = max(int(os.getenv("ACTIVE_STREAK_HARD_CAP", "12")), self.no_improvement_retire_streak)
        self.min_active_bots = max(int(min_active_bots), 0)
        self.correlation_prune_threshold = float(correlation_prune_threshold)
        self.signal_group_weight_target = float(os.getenv("SIGNAL_GROUP_WEIGHT_TARGET", "0.78"))
        self.walk_forward_fail_penalty = float(os.getenv("WALK_FORWARD_FAIL_PENALTY", "0.82"))
        self.walk_forward_min_forward_mean = float(os.getenv("WALK_FORWARD_MIN_FORWARD_MEAN", "0.51"))
        self.graduation_gate_enabled = os.getenv("MASTER_GRADUATION_GATE_ENABLED", "1").strip() == "1"
        self.graduation_min_runs = max(int(os.getenv("GRADUATION_MIN_RUNS", "24")), 1)
        self.graduation_min_forward_mean = float(os.getenv("GRADUATION_MIN_FORWARD_MEAN", "0.52"))
        self.graduation_min_delta = float(os.getenv("GRADUATION_MIN_DELTA", "-0.02"))
        self.min_trading_quality_score = float(os.getenv("MASTER_MIN_TRADING_QUALITY_SCORE", "0.50"))
        self.trading_quality_weight = min(max(float(os.getenv("MASTER_TRADING_QUALITY_WEIGHT", "0.35")), 0.0), 0.8)
        self.freeze_bot_count_enabled = os.getenv("MASTER_FREEZE_BOT_COUNT", "1").strip() == "1"
        self.strict_live_pass_only = os.getenv("MASTER_STRICT_LIVE_PASS_ONLY", "1").strip() == "1"

        self.prev_accuracy_by_bot = self._load_previous_accuracy_map()
        self.prev_best_accuracy_by_bot = self._load_previous_best_accuracy_map()
        self.prev_streak_by_bot = self._load_previous_streak_map()
        self.prev_status_by_bot = self._load_previous_status_map()
        self.prev_known_bot_ids = set(self.prev_status_by_bot.keys())
        self.walk_forward_map = self._load_walk_forward_map()
        self.correlation_map = self._load_decision_correlation_map()

    def train_from_outcomes(self) -> Dict[str, object]:
        outcomes = self._load_outcomes()
        statuses = self._evaluate_statuses(outcomes)
        statuses = self._enforce_bucket_diversity(statuses)
        statuses = self._enforce_min_active_bots(statuses)
        statuses = self._apply_correlation_pruning(statuses)
        statuses = self._enforce_active_streak_cap(statuses)
        statuses = self._assign_weights(statuses)
        payload = self._build_registry_payload(statuses)
        self._save_registry(payload)
        return payload

    def _load_outcomes(self) -> List[BotOutcome]:
        if not os.path.exists(self.logs_dir):
            return []

        outcomes: List[BotOutcome] = []
        for name in sorted(os.listdir(self.logs_dir)):
            if not name.startswith("brain_refinery_") or not name.endswith(".json"):
                continue

            path = os.path.join(self.logs_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue

            model_path = obj.get("model_path")
            metrics = obj.get("metrics") or {}
            bot_id = self._extract_bot_id_from_name(name)

            outcomes.append(
                BotOutcome(
                    bot_id=bot_id,
                    log_file=path,
                    model_path=model_path,
                    test_accuracy=self._as_float(metrics.get("test_accuracy")),
                    best_val_f1=self._as_float(metrics.get("best_val_f1")),
                    macro_f1=self._as_float(metrics.get("macro_f1") or metrics.get("test_macro_f1")),
                    final_val_loss=self._as_float(metrics.get("final_val_loss")),
                    max_drawdown=self._as_float(metrics.get("max_drawdown") or metrics.get("test_max_drawdown") or metrics.get("paper_max_drawdown")),
                    raw_metrics=metrics,
                )
            )

        latest_by_bot: Dict[str, BotOutcome] = {}
        for o in outcomes:
            latest_by_bot[o.bot_id] = o

        return list(latest_by_bot.values())

    def _load_previous_status_map(self) -> Dict[str, Dict[str, object]]:
        if not os.path.exists(self.registry_path):
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            return {}

        out: Dict[str, Dict[str, object]] = {}
        for row in prev.get("sub_bots", []):
            bot_id = str(row.get("bot_id", "")).strip()
            if bot_id:
                out[bot_id] = row
        return out

    def _load_previous_accuracy_map(self) -> Dict[str, float]:
        if not os.path.exists(self.registry_path):
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            return {}

        out: Dict[str, float] = {}
        for row in prev.get("sub_bots", []):
            bot_id = str(row.get("bot_id", "")).strip()
            if not bot_id:
                continue
            acc = self._as_float(row.get("test_accuracy"))
            if acc is not None:
                out[bot_id] = acc
        return out

    def _load_previous_best_accuracy_map(self) -> Dict[str, float]:
        if not os.path.exists(self.registry_path):
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            return {}

        out: Dict[str, float] = {}
        for row in prev.get("sub_bots", []):
            bot_id = str(row.get("bot_id", "")).strip()
            if not bot_id:
                continue
            prev_best = self._as_float(row.get("previous_best_accuracy"))
            if prev_best is None:
                prev_best = self._as_float(row.get("test_accuracy"))
            if prev_best is not None:
                out[bot_id] = prev_best
        return out

    def _load_previous_streak_map(self) -> Dict[str, int]:
        if not os.path.exists(self.registry_path):
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            return {}

        out: Dict[str, int] = {}
        for row in prev.get("sub_bots", []):
            bot_id = str(row.get("bot_id", "")).strip()
            if not bot_id:
                continue
            try:
                out[bot_id] = max(int(row.get("no_improvement_streak", 0)), 0)
            except Exception:
                out[bot_id] = 0
        return out

    def _load_walk_forward_map(self) -> Dict[str, Dict[str, object]]:
        path = os.path.join(self.project_root, "governance", "walk_forward", "walk_forward_latest.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            return {}
        bots = obj.get("bots", {})
        return bots if isinstance(bots, dict) else {}

    def _is_graduated(self, wf: Dict[str, object]) -> tuple[bool, str]:
        if not self.graduation_gate_enabled:
            return True, "disabled"

        runs = 0
        try:
            runs = int(wf.get("runs", 0) or 0)
        except Exception:
            runs = 0
        if runs < self.graduation_min_runs:
            return False, f"runs<{self.graduation_min_runs}"

        fwd = self._as_float(wf.get("forward_mean"))
        if fwd is not None and fwd < self.graduation_min_forward_mean:
            return False, f"forward_mean<{self.graduation_min_forward_mean:.3f}"

        delta = self._as_float(wf.get("delta"))
        if delta is not None and delta < self.graduation_min_delta:
            return False, f"delta<{self.graduation_min_delta:.3f}"

        status = str(wf.get("status") or "").lower()
        if status and status not in {"pass", ""}:
            return False, f"status={status}"

        return True, "ok"

    def _load_decision_correlation_map(self) -> Dict[tuple[str, str], float]:
        corr: Dict[tuple[str, str], float] = {}
        files = []
        decision_root = os.path.join(self.project_root, "decision_explanations")
        if os.path.isdir(decision_root):
            for sub in sorted(os.listdir(decision_root)):
                if not sub.startswith("shadow"):
                    continue
                d = os.path.join(decision_root, sub)
                if not os.path.isdir(d):
                    continue
                cands = sorted([os.path.join(d, x) for x in os.listdir(d) if x.startswith("decision_explanations_") and x.endswith(".jsonl")])
                if cands:
                    files.append(cands[-1])

        if not files:
            return corr

        series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=4000))

        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = deque(f, maxlen=30000)
            except Exception:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                strat = str(row.get("strategy") or "")
                if not strat.startswith("brain_refinery_"):
                    continue
                action = str(row.get("action") or "HOLD").upper()
                v = 1.0 if action == "BUY" else (-1.0 if action == "SELL" else 0.0)
                series[strat].append(v)

        bots = sorted(series.keys())
        for i in range(len(bots)):
            bi = bots[i]
            xi = list(series[bi])
            if len(xi) < 200:
                continue
            for j in range(i + 1, len(bots)):
                bj = bots[j]
                xj = list(series[bj])
                n = min(len(xi), len(xj))
                if n < 200:
                    continue
                a = xi[-n:]
                b = xj[-n:]
                ma = sum(a) / n
                mb = sum(b) / n
                va = sum((u - ma) ** 2 for u in a)
                vb = sum((u - mb) ** 2 for u in b)
                if va <= 1e-9 or vb <= 1e-9:
                    continue
                cov = sum((a[k] - ma) * (b[k] - mb) for k in range(n))
                c = cov / ((va ** 0.5) * (vb ** 0.5) + 1e-12)
                corr[(bi, bj)] = c
                corr[(bj, bi)] = c

        return corr

    @staticmethod
    def _bucket(bot_id: str, bot_role: str) -> str:
        name = (bot_id or "").lower()
        if bot_role == "infrastructure_sub_bot":
            if any(t in name for t in ("risk", "drawdown", "budget", "allocator", "sentinel")):
                return "risk"
            return "infra"
        if any(t in name for t in ("macro", "event", "calendar", "pmi", "ism", "pce", "cpi", "fomc", "labor", "rates", "dividend", "yield", "payout", "compounder", "bond", "treasury", "duration", "credit_spread")):
            return "macro"
        if any(t in name for t in ("mean", "revert", "bollinger", "vwap", "keltner")):
            return "mean_revert"
        if any(t in name for t in ("shock", "flash", "crash", "garch", "vol")):
            return "shock"
        if any(t in name for t in ("trend", "breakout", "momentum", "dmi", "donchian")):
            return "trend"
        return "signal_other"

    def _enforce_bucket_diversity(self, statuses: List[BotStatus]) -> List[BotStatus]:
        bucket_min = {
            "trend": 3,
            "mean_revert": 3,
            "shock": 3,
            "macro": 2,
            "risk": 3,
        }
        by_bucket: Dict[str, List[BotStatus]] = defaultdict(list)
        for s in statuses:
            by_bucket[self._bucket(s.bot_id, s.bot_role)].append(s)

        for bucket, need in bucket_min.items():
            active = [x for x in by_bucket.get(bucket, []) if x.active]
            if len(active) >= need:
                continue
            candidates = sorted(
                [x for x in by_bucket.get(bucket, []) if not x.active and x.candidate_test_accuracy is not None],
                key=lambda x: (x.candidate_quality_score, x.candidate_test_accuracy or 0.0),
                reverse=True,
            )
            for c in candidates[: max(need - len(active), 0)]:
                c.active = True
                c.reason = f"bucket_diversity_{bucket}"
                c.deleted_from_rotation = False
                c.delete_reason = ""
                if c.test_accuracy is None and c.candidate_test_accuracy is not None:
                    c.test_accuracy = c.candidate_test_accuracy
                if c.quality_score <= 0:
                    c.quality_score = max(c.candidate_quality_score, 0.01)
                c.preference_score = max(self._preference_score(c.test_accuracy or 0.5), 1e-6)
        return statuses

    def _apply_correlation_pruning(self, statuses: List[BotStatus]) -> List[BotStatus]:
        active = [s for s in statuses if s.active and s.bot_role == "signal_sub_bot"]
        if len(active) < 2 or not self.correlation_map:
            return statuses

        threshold = self.correlation_prune_threshold
        to_disable = set()
        ranked = sorted(active, key=lambda x: (x.quality_score, x.test_accuracy or 0.0), reverse=True)

        for i in range(len(ranked)):
            a = ranked[i]
            if a.bot_id in to_disable:
                continue
            for j in range(i + 1, len(ranked)):
                b = ranked[j]
                if b.bot_id in to_disable:
                    continue
                c = self.correlation_map.get((a.bot_id, b.bot_id))
                if c is None or c < threshold:
                    continue
                loser = b if (a.quality_score >= b.quality_score) else a
                # keep minimum active floor intact
                remaining = sum(1 for s in statuses if s.active and s.bot_id not in to_disable)
                if remaining <= self.min_active_bots:
                    continue
                to_disable.add(loser.bot_id)

        if to_disable:
            for s in statuses:
                if s.bot_id in to_disable:
                    s.active = False
                    s.weight = 0.0
                    s.reason = f"correlation_pruned_gt_{threshold:.2f}"

        return statuses


    def _evaluate_statuses(self, outcomes: List[BotOutcome]) -> List[BotStatus]:
        statuses: List[BotStatus] = []

        for o in outcomes:
            candidate_acc = o.test_accuracy
            candidate_quality = self._quality_score(candidate_acc, o.best_val_f1, o.macro_f1, o.final_val_loss, o.max_drawdown, o.raw_metrics)

            prev_acc = self.prev_accuracy_by_bot.get(o.bot_id)
            prev_best = self.prev_best_accuracy_by_bot.get(o.bot_id)
            prev_row = self.prev_status_by_bot.get(o.bot_id, {})
            prev_streak = self.prev_streak_by_bot.get(o.bot_id, 0)
            prev_candidate_log_file = str(prev_row.get("candidate_log_file") or "")

            prev_deleted = bool(prev_row.get("deleted_from_rotation", False))
            if prev_deleted:
                statuses.append(
                    BotStatus(
                        bot_id=o.bot_id,
                        bot_role=self._infer_bot_role(o.bot_id),
                        active=False,
                        reason=str(prev_row.get("reason") or "deleted_from_rotation"),
                        weight=0.0,
                        preference_score=0.0,
                        quality_score=float(prev_row.get("quality_score") or 0.0),
                        test_accuracy=self._as_float(prev_row.get("test_accuracy")),
                        candidate_test_accuracy=candidate_acc,
                        candidate_quality_score=candidate_quality,
                        previous_best_accuracy=prev_best,
                        no_improvement_streak=prev_streak,
                        deleted_from_rotation=True,
                        delete_reason=str(prev_row.get("delete_reason") or "deleted_from_rotation"),
                        promoted=False,
                        promotion_reason="rotation_deleted",
                        model_path=prev_row.get("model_path") or o.model_path,
                        log_file=prev_row.get("log_file") or o.log_file,
                    )
                )
                continue

            if candidate_acc is None:
                statuses.append(
                    BotStatus(
                        bot_id=o.bot_id,
                        bot_role=self._infer_bot_role(o.bot_id),
                        active=False,
                        reason="no_classification_accuracy",
                        weight=0.0,
                        preference_score=0.0,
                        quality_score=0.0,
                        test_accuracy=None,
                        candidate_test_accuracy=None,
                        candidate_quality_score=0.0,
                        previous_best_accuracy=prev_best,
                        no_improvement_streak=prev_streak,
                        deleted_from_rotation=False,
                        delete_reason="",
                        promoted=False,
                        promotion_reason="no_candidate_accuracy",
                        model_path=o.model_path,
                        log_file=o.log_file,
                    )
                )
                continue

            is_new_bot = o.bot_id not in self.prev_known_bot_ids
            if self.freeze_bot_count_enabled and is_new_bot:
                statuses.append(
                    BotStatus(
                        bot_id=o.bot_id,
                        bot_role=self._infer_bot_role(o.bot_id),
                        active=False,
                        reason="frozen_new_bot_count",
                        weight=0.0,
                        preference_score=0.0,
                        quality_score=candidate_quality,
                        test_accuracy=candidate_acc,
                        candidate_test_accuracy=candidate_acc,
                        candidate_quality_score=candidate_quality,
                        previous_best_accuracy=prev_best,
                        no_improvement_streak=prev_streak,
                        deleted_from_rotation=False,
                        delete_reason="",
                        promoted=False,
                        promotion_reason="frozen_new_bot_count",
                        model_path=o.model_path,
                        log_file=o.log_file,
                    )
                )
                continue

            promoted = True
            promotion_reason = "promoted_new_model"
            effective_acc = candidate_acc
            effective_quality = candidate_quality
            effective_model_path = o.model_path
            effective_log_file = o.log_file

            prev_quality = self._as_float(prev_row.get("quality_score"))
            if prev_quality is not None and candidate_quality < (prev_quality + (0.5 * self.promotion_margin)):
                promoted = False
                promotion_reason = f"quality_gate_hold_prev_plus_{self.promotion_margin:.3f}"
                effective_acc = prev_acc
                effective_quality = self._as_float(prev_row.get("quality_score")) or candidate_quality
                effective_model_path = prev_row.get("model_path") or o.model_path
                effective_log_file = prev_row.get("log_file") or o.log_file

            baseline_best = prev_best if prev_best is not None else prev_acc
            improved = False
            candidate_is_new = (o.log_file != prev_candidate_log_file)
            if baseline_best is None:
                improved = True
                updated_best = candidate_acc
                streak = 0
            elif candidate_acc > (baseline_best + self.promotion_margin):
                improved = True
                updated_best = candidate_acc
                streak = 0
            else:
                updated_best = baseline_best
                # Only increment streak on a genuinely new retrain candidate.
                streak = (prev_streak + 1) if candidate_is_new else prev_streak

            wf = self.walk_forward_map.get(o.bot_id, {})
            wf_status = str(wf.get("status") or "").lower()
            wf_forward = self._as_float(wf.get("forward_mean"))
            wf_tq = self._as_float(wf.get("trading_quality_score"))
            graduated, grad_reason = self._is_graduated(wf)
            if wf_status == "fail" or (wf_forward is not None and wf_forward < self.walk_forward_min_forward_mean):
                effective_quality *= self.walk_forward_fail_penalty

            if wf_tq is not None:
                tq = self._clamp(wf_tq)
                w = self.trading_quality_weight
                effective_quality = ((1.0 - w) * effective_quality) + (w * tq)

            pref = self._preference_score(effective_acc)
            deleted_from_rotation = False
            delete_reason = ""

            if self.graduation_gate_enabled and (not graduated):
                active = False
                reason = f"graduation_hold:{grad_reason}"
            elif self.strict_live_pass_only and wf_status != "pass":
                active = False
                reason = f"walk_forward_{wf_status}_live_hold"
            elif wf_status == "fail" and (wf_forward is not None and wf_forward < self.walk_forward_min_forward_mean):
                active = False
                reason = "walk_forward_fail"
            elif wf_tq is not None and wf_tq < self.min_trading_quality_score:
                active = False
                reason = f"trading_quality_below_{self.min_trading_quality_score:.2f}"
            elif streak >= self.no_improvement_retire_streak and not improved:
                active = False
                deleted_from_rotation = True
                delete_reason = f"deleted_no_improvement_{self.no_improvement_retire_streak}_retrainings"
                reason = delete_reason
            elif prev_acc is not None and effective_acc < (prev_acc - self.decay_guard_drop):
                active = False
                reason = f"decay_guard_drop_{self.decay_guard_drop:.2f}"
            elif effective_acc < self.deactivate_below:
                active = False
                reason = f"accuracy_below_{self.deactivate_below:.2f}"
            elif effective_acc < self.quality_floor:
                active = False
                reason = f"accuracy_below_quality_floor_{self.quality_floor:.2f}"
            else:
                active = True
                reason = "within_operating_band" if self.preferred_low <= effective_acc <= self.preferred_high else "active_outside_band"

            statuses.append(
                BotStatus(
                    bot_id=o.bot_id,
                    bot_role=self._infer_bot_role(o.bot_id),
                    active=active,
                    reason=reason,
                    weight=0.0,
                    preference_score=pref,
                    quality_score=effective_quality,
                    test_accuracy=effective_acc,
                    candidate_test_accuracy=candidate_acc,
                    candidate_quality_score=candidate_quality,
                    candidate_log_file=o.log_file,
                    previous_best_accuracy=updated_best,
                    no_improvement_streak=streak,
                    deleted_from_rotation=deleted_from_rotation,
                    delete_reason=delete_reason,
                    promoted=promoted,
                    promotion_reason=promotion_reason,
                    model_path=effective_model_path,
                    log_file=effective_log_file,
                )
            )

        return statuses

    def _enforce_min_active_bots(self, statuses: List[BotStatus]) -> List[BotStatus]:
        active_count = sum(1 for s in statuses if s.active)
        if active_count >= self.min_active_bots:
            return statuses

        needed = self.min_active_bots - active_count

        def rank_key(s: BotStatus):
            acc = s.candidate_test_accuracy if s.candidate_test_accuracy is not None else -1.0
            return (s.candidate_quality_score, acc, s.quality_score)

        def _eligible_for_floor_override(s: BotStatus) -> bool:
            reason = str(s.reason or "")
            if reason.startswith("graduation_hold:"):
                return False
            if self.strict_live_pass_only and reason.startswith("walk_forward_"):
                return False
            if self.freeze_bot_count_enabled and reason.startswith("frozen_new_bot_count"):
                return False
            return True

        candidates = sorted(
            [
                s
                for s in statuses
                if (not s.active)
                and (not s.deleted_from_rotation)
                and (s.candidate_test_accuracy is not None)
                and _eligible_for_floor_override(s)
            ],
            key=rank_key,
            reverse=True,
        )

        revived_deleted = sorted(
            [
                s
                for s in statuses
                if (not s.active)
                and s.deleted_from_rotation
                and (s.candidate_test_accuracy is not None)
                and (not str(s.delete_reason or "").startswith("active_streak_cap_"))
                and _eligible_for_floor_override(s)
            ],
            key=rank_key,
            reverse=True,
        )

        promoted: List[BotStatus] = []
        for st in candidates:
            if len(promoted) >= needed:
                break
            promoted.append(st)

        if len(promoted) < needed:
            for st in revived_deleted:
                if len(promoted) >= needed:
                    break
                promoted.append(st)

        for st in promoted:
            st.active = True
            st.reason = f"min_active_floor_override_{self.min_active_bots}"
            st.deleted_from_rotation = False
            st.delete_reason = ""
            if st.test_accuracy is None and st.candidate_test_accuracy is not None:
                st.test_accuracy = st.candidate_test_accuracy
            if st.quality_score <= 0.0:
                st.quality_score = max(st.candidate_quality_score, 0.01)
            st.preference_score = max(self._preference_score(st.test_accuracy or 0.50), 1e-6)

        return statuses


    def _enforce_active_streak_cap(self, statuses: List[BotStatus]) -> List[BotStatus]:
        cap = self.max_active_no_improvement_streak
        if cap <= 0:
            return statuses

        active_now = [s for s in statuses if s.active]
        remaining = len(active_now)

        candidates = sorted(
            [s for s in active_now if s.no_improvement_streak >= cap],
            key=lambda s: (s.quality_score, -(s.test_accuracy or 0.0), -s.no_improvement_streak),
        )

        for s in candidates:
            # Never breach minimum active floor; stability takes priority over pruning.
            if (remaining - 1) < self.min_active_bots:
                break
            s.active = False
            s.weight = 0.0
            s.deleted_from_rotation = True
            s.delete_reason = f"active_streak_cap_{cap}"
            s.reason = s.delete_reason
            s.promotion_reason = "rotation_deleted"
            remaining -= 1

        return statuses


    def _assign_weights(self, statuses: List[BotStatus]) -> List[BotStatus]:
        active = [s for s in statuses if s.active]
        total = sum(max(s.preference_score, 1e-8) for s in active)

        if total <= 0.0:
            return statuses

        for s in statuses:
            if not s.active:
                s.weight = 0.0
            else:
                s.weight = max(s.preference_score, 1e-8) / total

        flash = [s for s in active if "flash_crash" in s.bot_id]
        if flash:
            lead = max(flash, key=lambda x: x.weight)
            cap = min(max(self.flash_crash_weight_cap, 0.0), 1.0)
            if lead.weight > cap:
                excess = lead.weight - cap
                lead.weight = cap

                others = [s for s in active if s is not lead]
                others_total = sum(s.weight for s in others)
                if others_total > 0:
                    for s in others:
                        s.weight += excess * (s.weight / others_total)

        active_total = sum(s.weight for s in active)
        if active_total > 0:
            for s in active:
                s.weight /= active_total

        signal = [s for s in active if s.bot_role == "signal_sub_bot"]
        infra = [s for s in active if s.bot_role == "infrastructure_sub_bot"]
        if signal and infra:
            sig_target = min(max(self.signal_group_weight_target, 0.50), 0.90)
            inf_target = 1.0 - sig_target
            sig_sum = sum(s.weight for s in signal)
            inf_sum = sum(s.weight for s in infra)
            if sig_sum > 0 and inf_sum > 0:
                for s in signal:
                    s.weight = (s.weight / sig_sum) * sig_target
                for s in infra:
                    s.weight = (s.weight / inf_sum) * inf_target

        total2 = sum(s.weight for s in active)
        if total2 > 0:
            for s in active:
                s.weight /= total2

        return statuses

    def _build_registry_payload(self, statuses: List[BotStatus]) -> Dict[str, object]:
        active_count = sum(1 for s in statuses if s.active)
        inactive_count = len(statuses) - active_count
        deleted_count = sum(1 for s in statuses if s.deleted_from_rotation)

        top_active = sorted([s for s in statuses if s.active], key=lambda x: x.weight, reverse=True)

        return {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "master_policy": {
                "preferred_band": [self.preferred_low, self.preferred_high],
                "deactivate_below": self.deactivate_below,
                "quality_floor": self.quality_floor,
                "flash_crash_weight_cap": self.flash_crash_weight_cap,
                "decay_guard_drop": self.decay_guard_drop,
                "promotion_margin": self.promotion_margin,
                "no_improvement_retire_streak": self.no_improvement_retire_streak,
                "max_active_no_improvement_streak": self.max_active_no_improvement_streak,
                "min_active_bots": self.min_active_bots,
                "graduation_gate_enabled": self.graduation_gate_enabled,
                "graduation_min_runs": self.graduation_min_runs,
                "graduation_min_forward_mean": self.graduation_min_forward_mean,
                "graduation_min_delta": self.graduation_min_delta,
                "min_trading_quality_score": self.min_trading_quality_score,
                "trading_quality_weight": self.trading_quality_weight,
                "freeze_bot_count_enabled": self.freeze_bot_count_enabled,
                "strict_live_pass_only": self.strict_live_pass_only,
                "quality_score_formula": {
                    "accuracy_weight": 0.65,
                    "val_f1_weight": 0.25,
                    "val_loss_weight": 0.10,
                },
            },
            "summary": {
                "total_bots": len(statuses),
                "active_bots": active_count,
                "inactive_bots": inactive_count,
                "deleted_from_rotation": deleted_count,
                "active_signal_sub_bots": sum(1 for s in statuses if s.active and s.bot_role == "signal_sub_bot"),
                "active_infrastructure_sub_bots": sum(1 for s in statuses if s.active and s.bot_role == "infrastructure_sub_bot"),
                "inactive_signal_sub_bots": sum(1 for s in statuses if (not s.active) and s.bot_role == "signal_sub_bot"),
                "inactive_infrastructure_sub_bots": sum(1 for s in statuses if (not s.active) and s.bot_role == "infrastructure_sub_bot"),
                "promoted_models": sum(1 for s in statuses if s.promoted),
                "held_previous_models": sum(1 for s in statuses if not s.promoted),
                "top_active": [
                    {
                        "bot_id": s.bot_id,
                        "bot_role": s.bot_role,
                        "weight": round(s.weight, 6),
                        "test_accuracy": s.test_accuracy,
                        "quality_score": round(s.quality_score, 6),
                    }
                    for s in top_active[:5]
                ],
            },
            "sub_bots": [asdict(s) for s in sorted(statuses, key=lambda x: x.bot_id)],
        }

    def _save_registry(self, payload: Dict[str, object]) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _quality_score(
        self,
        test_accuracy: Optional[float],
        best_val_f1: Optional[float],
        macro_f1: Optional[float],
        final_val_loss: Optional[float],
        max_drawdown: Optional[float],
        raw_metrics: Optional[Dict[str, float]] = None,
    ) -> float:
        if test_accuracy is None:
            return 0.0

        acc_component = self._clamp((test_accuracy - 0.45) / 0.25)
        f1_component = self._clamp(best_val_f1 if best_val_f1 is not None else acc_component)
        macro_component = self._clamp(macro_f1 if macro_f1 is not None else f1_component)
        loss_component = 0.5
        if final_val_loss is not None:
            loss_component = self._clamp((1.2 - final_val_loss) / 1.2)

        dd_component = 0.5
        if max_drawdown is not None:
            dd_abs = abs(float(max_drawdown))
            dd_component = self._clamp(1.0 - min(dd_abs / 0.20, 1.0))

        no_trade_component = 0.5
        if raw_metrics:
            nt = self._as_float(raw_metrics.get("neutral_f1") or raw_metrics.get("hold_f1") or raw_metrics.get("flat_f1"))
            if nt is not None:
                no_trade_component = self._clamp(nt)

        return (
            0.45 * acc_component
            + 0.18 * f1_component
            + 0.10 * macro_component
            + 0.10 * loss_component
            + 0.10 * dd_component
            + 0.07 * no_trade_component
        )

    def _preference_score(self, accuracy: float) -> float:
        center = (self.preferred_low + self.preferred_high) / 2.0
        sigma = max((self.preferred_high - self.preferred_low) / 2.0, 1e-3)

        gaussian = math.exp(-0.5 * ((accuracy - center) / sigma) ** 2)
        in_band_bonus = 1.25 if self.preferred_low <= accuracy <= self.preferred_high else 1.0
        viability = max(accuracy - 0.50, 0.0)

        return in_band_bonus * (0.8 * gaussian + 0.2 * viability)

    @staticmethod
    def _infer_bot_role(bot_id: str) -> str:
        name = bot_id.lower()
        infra_tokens = (
            "layer",
            "sentinel",
            "allocator",
            "pruner",
            "controller",
            "router",
            "threshold",
            "reliability",
            "penalty",
            "budget",
            "filter",
            "drift",
            "champion",
            "challenger",
            "calibrator",
            "meta_ranker",
            "cost_aware",
            "data_quality",
        )
        if any(tok in name for tok in infra_tokens):
            return "infrastructure_sub_bot"
        return "signal_sub_bot"

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def _as_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_bot_id_from_name(name: str) -> str:
        base = name[:-5] if name.endswith(".json") else name
        return TIMESTAMP_SUFFIX_RE.sub("", base)
