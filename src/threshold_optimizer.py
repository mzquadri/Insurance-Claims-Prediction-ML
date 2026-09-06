"""
Business-driven threshold optimization for insurance claims prediction.
Selects the optimal decision threshold that maximizes expected business value.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ──────────────────────────────────────────────
# Cost-sensitive threshold optimization
# ──────────────────────────────────────────────
def compute_business_value(y_true, y_pred, cost_matrix: dict | None = None):
    """
    Compute expected business value given a cost matrix.

    Parameters
    ----------
    cost_matrix : dict
        Keys: 'tp_value', 'tn_value', 'fp_cost', 'fn_cost'
        - tp_value: Value of correctly identifying a claim (e.g., fraud savings)
        - tn_value: Value of correctly rejecting a non-claim
        - fp_cost: Cost of falsely flagging a non-claim (investigation cost)
        - fn_cost: Cost of missing a real claim (payout)
    """
    if cost_matrix is None:
        cost_matrix = {
            "tp_value": 500,  # Catching a claim early saves on processing
            "tn_value": 0,  # No action needed
            "fp_cost": -50,  # Unnecessary investigation cost
            "fn_cost": -1000,  # Missing a claim is expensive
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    value = (
        tp * cost_matrix["tp_value"]
        + tn * cost_matrix["tn_value"]
        + fp * cost_matrix["fp_cost"]
        + fn * cost_matrix["fn_cost"]
    )
    return value


def metrics_at_threshold(y_true, y_prob, threshold: float, method: str = "fixed"):
    """Score a fixed threshold on a given set.

    Separated from the search so that a threshold found on one set can be
    reported on another, which is the only honest way to report it.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "method": method,
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def find_optimal_threshold(
    y_true, y_prob, method: str = "f1", cost_matrix: dict | None = None
):
    """
    Find the optimal classification threshold.

    Parameters
    ----------
    method : str
        'f1' - maximize F1 score
        'youden' - maximize Youden's J statistic (sensitivity + specificity - 1)
        'business' - maximize business value using cost matrix
        'precision_at_recall' - find threshold for minimum recall of 0.9

    Returns
    -------
    optimal_threshold : float
    metrics_at_threshold : dict
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_threshold = 0.5
    best_score = -np.inf

    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if method == "f1":
            score = f1
        elif method == "youden":
            score = sensitivity + specificity - 1
        elif method == "business":
            score = compute_business_value(y_true, y_pred, cost_matrix)
        elif method == "precision_at_recall":
            score = precision if sensitivity >= 0.9 else -1
        else:
            raise ValueError(f"Unknown method: {method}")

        results.append(
            {
                "threshold": t,
                "score": score,
                "accuracy": accuracy,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "f1": f1,
            }
        )

        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"\nOptimal threshold ({method}): {best_threshold:.2f}")

    metrics = metrics_at_threshold(y_true, y_prob, best_threshold, method=method)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    return best_threshold, metrics, results


def plot_threshold_analysis(y_true, y_prob, save_dir: Path | None = None):
    """Plot threshold vs. various metrics."""
    save_dir = RESULTS_DIR if save_dir is None else save_dir
    thresholds = np.arange(0.01, 1.0, 0.01)
    sensitivities, specificities, precisions, f1s, accuracies = [], [], [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(prec)
        sens = sensitivities[-1]
        f1s.append(2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0)
        accuracies.append((tp + tn) / (tp + tn + fp + fn))

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Metrics vs threshold
    axes[0].plot(
        thresholds, sensitivities, "b-", linewidth=2, label="Sensitivity (Recall)"
    )
    axes[0].plot(thresholds, specificities, "r-", linewidth=2, label="Specificity")
    axes[0].plot(thresholds, precisions, "g-", linewidth=2, label="Precision")
    axes[0].plot(thresholds, f1s, "m-", linewidth=2, label="F1 Score")
    axes[0].plot(thresholds, accuracies, "k--", linewidth=1, label="Accuracy")

    best_f1_idx = np.argmax(f1s)
    axes[0].axvline(thresholds[best_f1_idx], color="gray", linestyle=":", alpha=0.7)
    axes[0].set_xlabel("Threshold", fontsize=12)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].set_title("Metrics vs. Decision Threshold", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall trade-off
    axes[1].plot(sensitivities, precisions, "b-", linewidth=2)
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title("Precision-Recall Trade-off", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Threshold analysis saved to {save_dir / 'threshold_analysis.png'}")


def run_threshold_optimization(model_path: str):
    """Run threshold optimization pipeline."""
    model = joblib.load(model_path)
    # Closed rather than left open. An NpzFile holds the archive handle until it
    # is closed, which on Windows blocks the directory from being removed.
    with np.load(RESULTS_DIR / "processed_data.npz") as archive:
        data = dict(archive.items())
    X_test, y_test = data["X_test"], data["y_test"]

    # The threshold is chosen against validation labels and then scored against
    # test labels. Choosing it on the test set and reporting the value there, as
    # this function used to, cannot come out worse than choosing it anywhere
    # else, because the search maximises the very quantity being reported.
    # src/benchmark.py measures the size of that over two hundred repartitions.
    if "X_validation" not in data:
        raise SystemExit(
            "processed_data.npz has no validation split. Rerun "
            "src/data_pipeline.py, which now produces one. A threshold selected "
            "on the test set cannot honestly be reported on the test set."
        )
    X_validation, y_validation = data["X_validation"], data["y_validation"]

    y_prob_validation = model.predict_proba(X_validation)[:, 1]
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=" * 50)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 50)
    print(f"Thresholds selected on {len(y_validation)} validation rows, "
          f"reported on {len(y_test)} test rows.")

    # Each threshold is selected on validation, then the metrics reported beside
    # it are recomputed on the test set at that fixed threshold.
    t_f1, _, _ = find_optimal_threshold(y_validation, y_prob_validation, method="f1")
    m_f1 = metrics_at_threshold(y_test, y_prob, t_f1)

    t_youden, _, _ = find_optimal_threshold(
        y_validation, y_prob_validation, method="youden")
    m_youden = metrics_at_threshold(y_test, y_prob, t_youden)

    t_biz, _, _ = find_optimal_threshold(
        y_validation, y_prob_validation, method="business")
    m_biz = metrics_at_threshold(y_test, y_prob, t_biz)

    # Comparison
    print(
        f"\n{'Method':<25} {'Threshold':<12} {'F1':<10} {'Sensitivity':<14} {'Specificity':<14}"
    )
    print("-" * 75)
    for name, t, m in [
        ("F1 Optimized", t_f1, m_f1),
        ("Youden's J", t_youden, m_youden),
        ("Business Value", t_biz, m_biz),
    ]:
        print(
            f"{name:<25} {t:<12.3f} {m['f1']:<10.4f} "
            f"{m['sensitivity']:<14.4f} {m['specificity']:<14.4f}"
        )

    # Persist the chosen thresholds. They are part of the decision rule, not a
    # by-product of it: a calibrated model plus a threshold is what makes a
    # prediction, and reproducing a reported number needs both.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"f1": t_f1, "youden": t_youden, "business": t_biz},
                RESULTS_DIR / "optimal_thresholds.pkl")
    print(f"\nThresholds saved to {RESULTS_DIR / 'optimal_thresholds.pkl'}")

    # Plot
    plot_threshold_analysis(y_test, y_prob)

    return {"f1": t_f1, "youden": t_youden, "business": t_biz}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold optimization")
    parser.add_argument(
        "--model_path", type=str, default=str(RESULTS_DIR / "calibrated_model.pkl")
    )
    args = parser.parse_args()

    run_threshold_optimization(args.model_path)
