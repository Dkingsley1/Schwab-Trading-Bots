import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.ops.restart_sanity_bundle as bundle


def test_restart_sanity_bundle_watcher_summary_marks_recent_file_fresh(tmp_path) -> None:
    status_path = tmp_path / "macro_auto_watch_status.json"
    status_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stream_state": "live",
                "resolved_video_url": "https://example.com/live",
                "youtube_channel_url": "https://example.com/channel",
                "media_ingest_triggered": True,
            }
        ),
        encoding="utf-8",
    )

    payload = bundle._watcher_summary(status_path, max_age_hours=6.0)

    assert payload["exists"] is True
    assert payload["fresh"] is True
    assert payload["stream_state"] == "live"
    assert payload["media_ingest_triggered"] is True
