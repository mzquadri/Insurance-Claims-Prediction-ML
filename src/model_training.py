"""
Model training module with cross-validation, hyperparameter tuning,
and model comparison for insurance claims prediction.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    classification_report,
    make_scorer,
)

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ──────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────
def get_models():
    """Return a dictionary of candidate models."""
    models = {
        "logistic_regression": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    if HAS_XGB:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if HAS_LGB:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    return models


# ──────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────
def cross_validate_models(X_train, y_train, models: dict = None, n_folds: int = 5):
    """
    Run stratified K-fold cross-validation on all models.

    Returns
    -------
    results : dict
        Model name -> dict of metric arrays.
    """
    if models is None:
        models = get_models()

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "neg_brier": make_scorer(
            brier_score_loss, needs_proba=True, greater_is_better=False
        ),
        "neg_log_loss": "neg_log_loss",
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        results[name] = {
            "accuracy": cv_results["test_accuracy"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "roc_auc": cv_results["test_roc_auc"].mean(),
            "roc_auc_std": cv_results["test_roc_auc"].std(),
            "brier_score": -cv_results["test_neg_brier"].mean(),
            "log_loss": -cv_results["test_neg_log_loss"].mean(),
        }

        print(
            f"  Accuracy:    {results[name]['accuracy']:.4f} (+/- {results[name]['accuracy_std']:.4f})"
        )
        print(
            f"  AUC-ROC:     {results[name]['roc_auc']:.4f} (+/- {results[name]['roc_auc_std']:.4f})"
        )
        print(f"  Brier Score: {results[name]['brier_score']:.4f}")
        print(f"  Log Loss:    {results[name]['log_loss']:.4f}")

    return results


# ──────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────
def train_best_model(X_train, y_train, X_test, y_test, model_name: str = "xgboost"):
    """Train the selected model on full training data and evaluate on test set."""
    models = get_models()

    if model_name not in models:
        available = list(models.keys())
        print(f"Model '{model_name}' not available. Using '{available[-1]}'")
        model_name = available[-1]

    model = models[model_name]
    print(f"\nTraining final model: {model_name}")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob),
    }

    print(f"\n{'=' * 50}")
    print(f"TEST SET RESULTS - {model_name.upper()}")
    print(f"{'=' * 50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    print(
        f"\n{classification_report(y_test, y_pred, target_names=['No Claim', 'Claim'])}"
    )

    # Save model and metrics
    RESULTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, RESULTS_DIR / "best_model.pkl")
    with open(RESULTS_DIR / "cv_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {RESULTS_DIR / 'best_model.pkl'}")
    return model, metrics


def load_processed_data():
    """Load preprocessed data from disk."""
    data = np.load(RESULTS_DIR / "processed_data.npz")
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train insurance claims models")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        help="Model to train: logistic_regression, random_forest, xgboost, lightgbm",
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_processed_data()

    # Cross-validate all models
    cv_results = cross_validate_models(X_train, y_train, n_folds=args.cv_folds)

    # Save CV comparison
    cv_df = pd.DataFrame(cv_results).T
    cv_df.to_csv(RESULTS_DIR / "model_comparison.csv")
    print(f"\nModel comparison saved to {RESULTS_DIR / 'model_comparison.csv'}")

    # Train best model
    model, metrics = train_best_model(
        X_train, y_train, X_test, y_test, model_name=args.model
    )
