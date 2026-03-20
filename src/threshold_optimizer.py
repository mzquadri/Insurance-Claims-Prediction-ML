"""
Business-driven threshold optimization for insurance claims prediction.
Selects the optimal decision threshold that maximizes expected business value.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ──────────────────────────────────────────────
# Cost-sensitive threshold optimization
# ──────────────────────────────────────────────
def compute_business_value(y_true, y_pred, cost_matrix: dict = None):
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


def find_optimal_threshold(
    y_true, y_prob, method: str = "f1", cost_matrix: dict = None
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

    # Metrics at optimal threshold
    y_pred_optimal = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()

    metrics = {
        "threshold": best_threshold,
        "method": method,
        "accuracy": accuracy_score(y_true, y_pred_optimal),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "f1": f1_score(y_true, y_pred_optimal),
    }

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    return best_threshold, metrics, results


def plot_threshold_analysis(y_true, y_prob, save_dir: Path = RESULTS_DIR):
    """Plot threshold vs. various metrics."""
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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
    data = np.load(RESULTS_DIR / "processed_data.npz")
    X_test, y_test = data["X_test"], data["y_test"]

    y_prob = model.predict_proba(X_test)[:, 1]

    print("=" * 50)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 50)

    # F1-based threshold
    t_f1, m_f1, _ = find_optimal_threshold(y_test, y_prob, method="f1")

    # Youden's J
    t_youden, m_youden, _ = find_optimal_threshold(y_test, y_prob, method="youden")

    # Business value
    t_biz, m_biz, _ = find_optimal_threshold(y_test, y_prob, method="business")

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
            f"{name:<25} {t:<12.3f} {m['f1']:<10.4f} {m['sensitivity']:<14.4f} {m['specificity']:<14.4f}"
        )

    # Plot
    plot_threshold_analysis(y_test, y_prob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold optimization")
    parser.add_argument(
        "--model_path", type=str, default=str(RESULTS_DIR / "calibrated_model.pkl")
    )
    args = parser.parse_args()

    run_threshold_optimization(args.model_path)
