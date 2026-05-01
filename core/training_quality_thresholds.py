from __future__ import annotations


# Shared training-quality thresholds used by supportability, coverage,
# and retrain gating so the floor cannot drift between control surfaces.
TARGET_TEST_ACCURACY_FLOOR = 0.75
STRONG_TEST_ACCURACY_FLOOR = TARGET_TEST_ACCURACY_FLOOR
STAGED_SUPPORT_RECOVERY_TEST_ACCURACY_FLOOR = TARGET_TEST_ACCURACY_FLOOR

STRONG_COVERAGE_QUALITY_FLOOR = 0.60
TARGET_QUALITY_SCORE_FLOOR = STRONG_COVERAGE_QUALITY_FLOOR
