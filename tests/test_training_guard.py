import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.training_guard import (
    check_confirmed_training_success,
    check_registry_row_state_before_deletion,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_snapshot_training_coverage(
    root: Path,
    *,
    timestamp_utc: str,
    all_snapshot_data_incorporated: bool = True,
    snapshot_raw_sql_ingest_ratio: float = 1.0,
    snapshot_cov_fill_ratio: float = 1.0,
    snapshot_feature_coverage_ratio: float = 1.0,
    reason: str = "ok",
) -> Path:
    path = root / "governance" / "health" / "snapshot_training_coverage_latest.json"
    _write_json(
        path,
        {
            "timestamp_utc": timestamp_utc,
            "all_snapshot_data_incorporated": bool(all_snapshot_data_incorporated),
            "snapshot_raw_sql_ingest_ratio": float(snapshot_raw_sql_ingest_ratio),
            "snapshot_cov_fill_ratio": float(snapshot_cov_fill_ratio),
            "snapshot_feature_coverage_ratio": float(snapshot_feature_coverage_ratio),
            "reason": reason,
        },
    )
    return path


class TrainingGuardTests(unittest.TestCase):
    def test_confirmed_training_success_marker_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / "governance" / "health" / "training_success_latest.json"
            marker_ts = datetime.now(timezone.utc)
            _write_json(
                marker,
                {
                    "timestamp_utc": marker_ts.isoformat(),
                    "confirmed_training_success": True,
                    "trained_count": 4,
                    "failure_count": 0,
                    "master_update_status": "updated_registry:4",
                    "reason": "ok",
                },
            )
            _write_snapshot_training_coverage(
                root,
                timestamp_utc=(marker_ts + timedelta(minutes=1)).isoformat(),
            )

            ok, reason, details = check_confirmed_training_success(
                project_root=str(root),
                marker_path=str(marker),
                scorecard_path=str(root / "governance" / "health" / "retrain_scorecard_latest.json"),
                max_age_hours=24.0,
            )

            self.assertTrue(ok)
            self.assertEqual(reason, "ok")
            self.assertEqual(details.get("source"), "training_success_marker")
            snap = details.get("snapshot_training_guard") if isinstance(details.get("snapshot_training_guard"), dict) else {}
            self.assertTrue(bool(snap.get("all_snapshot_data_incorporated", False)))

    def test_confirmed_training_success_marker_stale_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / "governance" / "health" / "training_success_latest.json"
            _write_json(
                marker,
                {
                    "timestamp_utc": (datetime.now(timezone.utc) - timedelta(hours=80)).isoformat(),
                    "confirmed_training_success": True,
                    "trained_count": 2,
                    "failure_count": 0,
                    "master_update_status": "updated_registry:2",
                    "reason": "ok",
                },
            )

            ok, reason, details = check_confirmed_training_success(
                project_root=str(root),
                marker_path=str(marker),
                scorecard_path=str(root / "governance" / "health" / "retrain_scorecard_latest.json"),
                max_age_hours=24.0,
            )

            self.assertFalse(ok)
            self.assertTrue(reason.startswith("stale_training_success_marker"))
            self.assertEqual(details.get("source"), "training_success_marker")

    def test_confirmed_training_success_scorecard_fallback_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scorecard = root / "governance" / "health" / "retrain_scorecard_latest.json"
            score_ts = datetime.now(timezone.utc)
            _write_json(
                scorecard,
                {
                    "timestamp_utc": score_ts.isoformat(),
                    "status_counts": {"trained": 3},
                    "failure_count": 0,
                    "master_update_status": "updated_registry:3",
                },
            )
            _write_snapshot_training_coverage(
                root,
                timestamp_utc=(score_ts + timedelta(minutes=2)).isoformat(),
            )

            ok, reason, details = check_confirmed_training_success(
                project_root=str(root),
                marker_path=str(root / "governance" / "health" / "training_success_latest.json"),
                scorecard_path=str(scorecard),
                max_age_hours=24.0,
            )

            self.assertTrue(ok)
            self.assertEqual(reason, "ok")
            self.assertEqual(details.get("source"), "retrain_scorecard")

    def test_confirmed_training_success_scorecard_failures_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scorecard = root / "governance" / "health" / "retrain_scorecard_latest.json"
            _write_json(
                scorecard,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "status_counts": {"trained": 3},
                    "failure_count": 1,
                    "master_update_status": "updated_registry:3",
                },
            )

            ok, reason, details = check_confirmed_training_success(
                project_root=str(root),
                marker_path=str(root / "governance" / "health" / "training_success_latest.json"),
                scorecard_path=str(scorecard),
                max_age_hours=24.0,
            )

            self.assertFalse(ok)
            self.assertTrue(reason.startswith("training_failures_present"))
            self.assertEqual(details.get("source"), "retrain_scorecard")

    def test_missing_snapshot_training_coverage_blocks_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / "governance" / "health" / "training_success_latest.json"
            _write_json(
                marker,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "confirmed_training_success": True,
                    "trained_count": 5,
                    "failure_count": 0,
                    "master_update_status": "updated_registry:5",
                    "reason": "ok",
                },
            )

            ok, reason, details = check_confirmed_training_success(
                project_root=str(root),
                marker_path=str(marker),
                scorecard_path=str(root / "governance" / "health" / "retrain_scorecard_latest.json"),
                max_age_hours=24.0,
            )

            self.assertFalse(ok)
            self.assertTrue(reason.startswith("snapshot_training_guard_failed:missing_snapshot_training_coverage_artifact"))
            self.assertEqual(details.get("source"), "training_success_marker")

    def test_snapshot_training_coverage_older_than_marker_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / "governance" / "health" / "training_success_latest.json"
            marker_ts = datetime.now(timezone.utc)
            _write_json(
                marker,
                {
                    "timestamp_utc": marker_ts.isoformat(),
                    "confirmed_training_success": True,
                    "trained_count": 5,
                    "failure_count": 0,
                    "master_update_status": "updated_registry:5",
                    "reason": "ok",
                },
            )
            _write_snapshot_training_coverage(
                root,
                timestamp_utc=(marker_ts - timedelta(minutes=2)).isoformat(),
            )

            ok, reason, details = check_confirmed_training_success(
                project_root=str(root),
                marker_path=str(marker),
                scorecard_path=str(root / "governance" / "health" / "retrain_scorecard_latest.json"),
                max_age_hours=24.0,
            )

            self.assertFalse(ok)
            self.assertTrue(reason.startswith("snapshot_training_guard_failed:snapshot_training_coverage_older_than_training_success"))
            self.assertEqual(details.get("source"), "training_success_marker")

    def test_snapshot_training_coverage_ratio_below_required_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / "governance" / "health" / "training_success_latest.json"
            marker_ts = datetime.now(timezone.utc)
            _write_json(
                marker,
                {
                    "timestamp_utc": marker_ts.isoformat(),
                    "confirmed_training_success": True,
                    "trained_count": 5,
                    "failure_count": 0,
                    "master_update_status": "updated_registry:5",
                    "reason": "ok",
                },
            )
            _write_snapshot_training_coverage(
                root,
                timestamp_utc=(marker_ts + timedelta(minutes=1)).isoformat(),
                all_snapshot_data_incorporated=False,
                snapshot_raw_sql_ingest_ratio=0.92,
                snapshot_cov_fill_ratio=1.0,
                snapshot_feature_coverage_ratio=1.0,
                reason="snapshot_raw_sql_ingest_ratio_below_required",
            )

            ok, reason, details = check_confirmed_training_success(
                project_root=str(root),
                marker_path=str(marker),
                scorecard_path=str(root / "governance" / "health" / "retrain_scorecard_latest.json"),
                max_age_hours=24.0,
            )

            self.assertFalse(ok)
            self.assertTrue(reason.startswith("snapshot_training_guard_failed:snapshot_training_not_fully_incorporated"))
            self.assertEqual(details.get("source"), "training_success_marker")

    def test_registry_row_state_before_deletion_blocks_invalid_flags(self) -> None:
        ok, reason, details = check_registry_row_state_before_deletion(
            {
                "bot_id": "v_1",
                "active": True,
                "deleted_from_rotation": True,
                "lifecycle_state": "deleted",
                "no_improvement_streak": 4,
            },
            min_streak=3,
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "invalid_state_deleted_and_active")
        self.assertEqual(details.get("bot_id"), "v_1")

    def test_registry_row_state_before_deletion_enforces_min_streak(self) -> None:
        ok, reason, details = check_registry_row_state_before_deletion(
            {
                "bot_id": "v_2",
                "active": False,
                "deleted_from_rotation": False,
                "lifecycle_state": "inactive",
                "no_improvement_streak": 1,
            },
            min_streak=3,
        )

        self.assertFalse(ok)
        self.assertTrue(reason.startswith("streak_below_threshold"))
        self.assertEqual(details.get("bot_id"), "v_2")

    def test_registry_row_state_before_deletion_passes_valid_row(self) -> None:
        ok, reason, details = check_registry_row_state_before_deletion(
            {
                "bot_id": "v_3",
                "active": False,
                "deleted_from_rotation": False,
                "lifecycle_state": "inactive",
                "no_improvement_streak": 5,
            },
            min_streak=3,
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertEqual(details.get("lifecycle_state"), "inactive")


if __name__ == "__main__":
    unittest.main()
