from core.master_bot import MasterBot


def test_quality_score_prefers_balanced_actionable_metrics_over_raw_accuracy(tmp_path) -> None:
    master = MasterBot(project_root=str(tmp_path), min_active_bots=0)

    inflated_raw_accuracy = master._quality_score(
        test_accuracy=0.92,
        best_val_f1=0.58,
        macro_f1=0.55,
        final_val_loss=0.48,
        max_drawdown=0.04,
        raw_metrics={
            "positive_rate": 0.86,
            "label_balance_score": 0.28,
            "acted_accuracy": 0.55,
            "acted_coverage": 0.12,
            "long_precision": 0.95,
            "short_precision": 0.25,
            "precision_balance_score": 0.2632,
            "accuracy_lift_over_majority": 0.06,
        },
    )

    balanced_actionable = master._quality_score(
        test_accuracy=0.63,
        best_val_f1=0.64,
        macro_f1=0.62,
        final_val_loss=0.54,
        max_drawdown=0.04,
        raw_metrics={
            "positive_rate": 0.52,
            "label_balance_score": 0.96,
            "acted_accuracy": 0.64,
            "acted_coverage": 0.31,
            "long_precision": 0.63,
            "short_precision": 0.61,
            "precision_balance_score": 0.9683,
            "accuracy_lift_over_majority": 0.11,
        },
    )

    assert balanced_actionable > inflated_raw_accuracy


def test_quality_score_rewards_accuracy_lift_over_majority_baseline(tmp_path) -> None:
    master = MasterBot(project_root=str(tmp_path), min_active_bots=0)

    weak_lift = master._quality_score(
        test_accuracy=0.61,
        best_val_f1=0.60,
        macro_f1=0.59,
        final_val_loss=0.56,
        max_drawdown=0.05,
        raw_metrics={
            "positive_rate": 0.60,
            "label_balance_score": 0.80,
            "acted_accuracy": 0.60,
            "acted_coverage": 0.24,
            "long_precision": 0.61,
            "short_precision": 0.58,
            "precision_balance_score": 0.9508,
            "accuracy_lift_over_majority": 0.01,
        },
    )

    strong_lift = master._quality_score(
        test_accuracy=0.61,
        best_val_f1=0.60,
        macro_f1=0.59,
        final_val_loss=0.56,
        max_drawdown=0.05,
        raw_metrics={
            "positive_rate": 0.50,
            "label_balance_score": 1.00,
            "acted_accuracy": 0.60,
            "acted_coverage": 0.24,
            "long_precision": 0.61,
            "short_precision": 0.58,
            "precision_balance_score": 0.9508,
            "accuracy_lift_over_majority": 0.11,
        },
    )

    assert strong_lift > weak_lift
