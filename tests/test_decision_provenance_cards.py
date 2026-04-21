import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import decision_provenance_cards as src


def test_decision_provenance_cards_parse_latest_decisions_logs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    first = project_root / "decision_explanations" / "shadow_aggressive_equities" / "latest_decisions.log"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text(
        "[Decision] mode=shadow_aggressive_equities status=DATA_ONLY_BLOCKED symbol=SPY action=BUY score=0.6120 threshold=0.5200 reasons=score_above_threshold | trend_support | bot_id=brain_refinery_v43 bot_role=signal_sub_bot gates=market_data_ok=PASS safety=market_data_only=1,execution_enabled=0\n",
        encoding="utf-8",
    )
    second = project_root / "decision_explanations" / "shadow_bond_equities" / "latest_decisions.log"
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_text(
        "[Decision] mode=shadow_bond_equities status=SHADOW_ONLY symbol=TLT action=HOLD score=0.5537 threshold=0.5950 reasons=options_specialist_signal | carry_filter | bot_id=options_specialist bot_role=options_sub_bot gates=market_data_ok=PASS safety=market_data_only=1,execution_enabled=0\n",
        encoding="utf-8",
    )

    payload = src.build_payload(project_root, limit=5)

    assert payload["overall_status"] == "ready"
    assert payload["card_count"] == 2
    assert payload["mode_count"] == 2
    top = payload["recent_cards"][0]
    assert top["symbol"] in {"SPY", "TLT"}
    assert top["bot_id"] in {"brain_refinery_v43", "options_specialist"}
    assert len(payload["review_sha256"]) == 64
