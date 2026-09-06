"""
Probability calibration for insurance claims prediction models.
Implements Platt scaling and isotonic regression to ensure
predicted probabilities align with true event frequencies.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
    y_true, probabilities_dict, n_bins: int = 10, save_dir: Path | None = None
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
    save_dir = RESULTS_DIR if save_dir is None else save_dir
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

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
    with np.load(RESULTS_DIR / "processed_data.npz") as archive:
        data = dict(archive.items())
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # Which calibrator wins is a decision made by looking at labels, so it is
    # made on validation. This function used to compare the three on the test set
    # and keep the winner, which is the same mistake as tuning on test.
    if "X_validation" not in data:
        raise SystemExit(
            "processed_data.npz has no validation split. Rerun "
            "src/data_pipeline.py, which now produces one. Choosing a calibrator "
            "on the test set and reporting its score there overstates it."
        )
    X_validation, y_validation = data["X_validation"], data["y_validation"]

    # Calibrators are fitted on training rows with internal folds. Nothing here
    # touches the test set until the winner has already been chosen.
    cal_sigmoid = calibrate_model(model, X_train, y_train, method="sigmoid")
    cal_isotonic = calibrate_model(model, X_train, y_train, method="isotonic")

    brier_validation = {
        "Platt Scaling": brier_score_loss(
            y_validation, cal_sigmoid.predict_proba(X_validation)[:, 1]),
        "Isotonic Regression": brier_score_loss(
            y_validation, cal_isotonic.predict_proba(X_validation)[:, 1]),
    }
    print(f"\nSelection on {len(y_validation)} validation rows "
          "(Brier - lower is better):")
    for label, value in brier_validation.items():
        print(f"  {label:<20} {value:.4f}")

    # Reported on the test set, at the choice already made.
    y_prob_uncalibrated = model.predict_proba(X_test)[:, 1]
    brier_uncalibrated = brier_score_loss(y_test, y_prob_uncalibrated)
    y_prob_platt = cal_sigmoid.predict_proba(X_test)[:, 1]
    brier_platt = brier_score_loss(y_test, y_prob_platt)
    y_prob_isotonic = cal_isotonic.predict_proba(X_test)[:, 1]
    brier_isotonic = brier_score_loss(y_test, y_prob_isotonic)

    print("\nCalibration Results (Brier Score - lower is better):")
    print(f"  Uncalibrated:       {brier_uncalibrated:.4f}")
    print(f"  Platt Scaling:      {brier_platt:.4f}")
    print(f"  Isotonic Regression: {brier_isotonic:.4f}")

    # The winner is whichever did better on validation, not on the test set.
    best_name = min(brier_validation, key=brier_validation.get)
    best_model = cal_sigmoid if best_name == "Platt Scaling" else cal_isotonic
    best_brier = brier_platt if best_name == "Platt Scaling" else brier_isotonic

    improvement = (brier_uncalibrated - best_brier) / brier_uncalibrated * 100
    print(f"\n  Chosen on validation: {best_name}")
    print(f"  Its test Brier: {best_brier:.4f} "
          f"({improvement:+.1f}% against uncalibrated)")
    if best_name != min(
            {"Platt Scaling": brier_platt, "Isotonic Regression": brier_isotonic},
            key=lambda k: {"Platt Scaling": brier_platt,
                           "Isotonic Regression": brier_isotonic}[k]):
        print("  Note: the other method scored better on the test set. Selecting "
              "on that basis is what this split exists to prevent.")

    # Save calibrated model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, RESULTS_DIR / "calibrated_model.pkl")

    # Plot
    plot_calibration_curve(
        y_test,
        {
            "Uncalibrated": y_prob_uncalibrated,
            "Platt Scaling": y_prob_platt,
            "Isotonic": y_prob_isotonic,
        },
        save_dir=RESULTS_DIR,
    )

    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate insurance claims model")
    parser.add_argument(
        "--model_path", type=str, default=str(RESULTS_DIR / "best_model.pkl")
    )
    args = parser.parse_args()

    run_calibration(args.model_path)
