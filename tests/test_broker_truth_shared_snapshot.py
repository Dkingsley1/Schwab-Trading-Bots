import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LIVE_PROJECT_ROOT = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot")
if not (PROJECT_ROOT / "core").exists() and str(LIVE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LIVE_PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import scripts.run_shadow_training_loop as loop


class _FailingTrader:
    client = object()

    def _live_fetch_accounts_payload(self):
        raise AssertionError("shared broker truth snapshot should have been reused")


def test_shared_broker_truth_snapshot_reuses_recent_cached_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    cache_path = loop._broker_truth_shared_snapshot_cache_path(str(tmp_path), "schwab")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "broker": "schwab",
                "owner_pid": os.getpid() + 1000,
                "fetched": {
                    "ok": False,
                    "error": "RuntimeError:http_status_403",
                    "soft_failure": True,
                    "soft_fail_streak": 1,
                    "soft_fail_grace": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    fetched = loop._shared_broker_truth_accounts_payload(trader=_FailingTrader(), broker="schwab")

    assert fetched["error"] == "RuntimeError:http_status_403"
    assert fetched["_shared_snapshot_cache_hit"] is True
    assert fetched["_shared_snapshot_cache_owner_pid"] != os.getpid()


def test_fetch_broker_truth_snapshot_suppresses_duplicate_soft_fail_alert_from_shared_cache(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    cache_path = loop._broker_truth_shared_snapshot_cache_path(str(tmp_path), "schwab")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "broker": "schwab",
                "owner_pid": os.getpid() + 1000,
                "fetched": {
                    "ok": False,
                    "error": "RuntimeError:http_status_403",
                    "soft_failure": True,
                    "soft_fail_streak": 1,
                    "soft_fail_grace": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = loop._fetch_broker_truth_snapshot(
        trader=_FailingTrader(),
        broker="schwab",
        simulate=False,
        iter_count=7,
        manual_payload={},
        manual_tolerance=1.0,
        previous_state={},
    )

    assert snapshot["ok"] is False
    assert snapshot["status"] == "transient_error"
    assert snapshot["soft_failure"] is True
    assert snapshot["shared_snapshot_cache_hit"] is True
    assert snapshot["alert_suppressed"] is True
