"""
Probability calibration for insurance claims prediction models.
Implements Platt scaling and isotonic regression to ensure
predicted probabilities align with true event frequencies.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def calibrate_model(model, X_train, y_train, method: str = "sigmoid", cv: int = 5):
    """
    Calibrate a trained classifier using cross-validation.

    Parameters
    ----------
    model : estimator
        Trained classifier.
    method : str
        'sigmoid' (Platt scaling) or 'isotonic' (isotonic regression).
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    calibrated_model : CalibratedClassifierCV
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


def plot_calibration_curve(
    y_true, probabilities_dict, n_bins: int = 10, save_dir: Path = RESULTS_DIR
):
    """
    Plot reliability diagrams for multiple models.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    probabilities_dict : dict
        {model_name: predicted_probabilities}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    for i, (name, y_prob) in enumerate(probabilities_dict.items()):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )

        brier = brier_score_loss(y_true, y_prob)

        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker="o",
            linewidth=2,
            color=colors[i % len(colors)],
            label=f"{name} (Brier={brier:.4f})",
        )

        ax2.hist(
            y_prob,
            bins=50,
            alpha=0.5,
            color=colors[i % len(colors)],
            label=name,
            edgecolor="black",
        )

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title("Calibration Curve (Reliability Diagram)", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Probability distribution
    ax2.set_xlabel("Predicted Probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Predicted Probabilities", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "calibration_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration plots saved to {save_dir / 'calibration_curves.png'}")


def run_calibration(model_path: str):
    """Load model, calibrate, and compare."""
    model = joblib.load(model_path)
    data = np.load(RESULTS_DIR / "processed_data.npz")
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # Uncalibrated probabilities
    y_prob_uncalibrated = model.predict_proba(X_test)[:, 1]
    brier_uncalibrated = brier_score_loss(y_test, y_prob_uncalibrated)

    # Platt scaling
    cal_sigmoid = calibrate_model(model, X_train, y_train, method="sigmoid")
    y_prob_platt = cal_sigmoid.predict_proba(X_test)[:, 1]
    brier_platt = brier_score_loss(y_test, y_prob_platt)

    # Isotonic regression
    cal_isotonic = calibrate_model(model, X_train, y_train, method="isotonic")
    y_prob_isotonic = cal_isotonic.predict_proba(X_test)[:, 1]
    brier_isotonic = brier_score_loss(y_test, y_prob_isotonic)

    print(f"\nCalibration Results (Brier Score - lower is better):")
    print(f"  Uncalibrated:       {brier_uncalibrated:.4f}")
    print(f"  Platt Scaling:      {brier_platt:.4f}")
    print(f"  Isotonic Regression: {brier_isotonic:.4f}")

    # Select best calibration method
    best_brier = min(brier_platt, brier_isotonic)
    if brier_platt <= brier_isotonic:
        best_model = cal_sigmoid
        best_name = "Platt Scaling"
    else:
        best_model = cal_isotonic
        best_name = "Isotonic Regression"

    improvement = (brier_uncalibrated - best_brier) / brier_uncalibrated * 100
    print(f"\n  Best method: {best_name} ({improvement:.1f}% improvement)")

    # Save calibrated model
    joblib.dump(best_model, RESULTS_DIR / "calibrated_model.pkl")

    # Plot
    plot_calibration_curve(
        y_test,
        {
            "Uncalibrated": y_prob_uncalibrated,
            "Platt Scaling": y_prob_platt,
            "Isotonic": y_prob_isotonic,
        },
    )

    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate insurance claims model")
    parser.add_argument(
        "--model_path", type=str, default=str(RESULTS_DIR / "best_model.pkl")
    )
    args = parser.parse_args()

    run_calibration(args.model_path)
