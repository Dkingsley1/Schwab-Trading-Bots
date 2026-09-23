"""Read the pinned native source directly; never accept an imported BUY override."""

from core.schd_bot_handoff import prepare_handoff
from core.schd_capture_store import decision_capture
from scripts.ops.schd_decision_rehearsal import DIRECTORY, local, read_json
from scripts.ops.schd_native_decision import native_packet, read_latest


def observe_handoff(root, plan, quote, *, action, now):
    row, receipt, scan = read_latest(root, now=now)
    if row is None:
        return {"ready": False, "blockers": scan["issues"], "receipt": {}}
    market = decision_capture(root, row)
    candidate = read_json(
        local(root / "governance/runtime/production_candidate_state.json", root=root)
    )
    packet = native_packet(row, receipt, scan, market, candidate, now=now)
    return prepare_handoff(
        plan,
        packet,
        quote,
        action=action,
        candidate_id=candidate.get("candidate_id"),
        now=now,
    )
