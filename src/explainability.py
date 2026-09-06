"""
SHAP-based model explainability for insurance claims prediction.
Provides global and local feature importance explanations.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def explain_global(model, X_train, feature_names, save_dir: Path | None = None):
    """
    Generate global SHAP explanations.
    Shows which features are most important across all predictions.
    """
    # Imported here rather than at module scope so this file can be read,
    # compiled and imported without shap installed. shap is optional, and only
    # the paths that actually compute attributions need it.
    import shap

    print("Computing SHAP values (this may take a moment)...")

    # Use TreeExplainer for tree-based models, KernelExplainer for others
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train[:500])
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
        shap_values = explainer.shap_values(X_train[:200])

    # Handle multi-output SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (claim) SHAP values

    save_dir = RESULTS_DIR if save_dir is None else save_dir
    save_dir.mkdir(exist_ok=True)

    # Summary plot (beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_train[: len(shap_values)],
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title("SHAP Feature Importance (Global)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {save_dir / 'shap_summary.png'}")

    # Bar plot (mean absolute SHAP)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_train[: len(shap_values)],
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title("Mean |SHAP Value| (Feature Importance)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar plot saved to {save_dir / 'shap_importance_bar.png'}")

    return shap_values, explainer


def explain_local(
    model,
    explainer,
    X_sample,
    feature_names,
    idx: int = 0,
    save_dir: Path | None = None,
):
    """
    Generate local SHAP explanation for a single prediction.
    Shows how each feature contributed to a specific prediction.
    """
    import shap

    try:
        shap_values = explainer.shap_values(X_sample[idx : idx + 1])
    except Exception:
        shap_values = explainer.shap_values(X_sample[idx : idx + 1])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(14, 4))
    shap.force_plot(
        explainer.expected_value
        if not isinstance(explainer.expected_value, list)
        else explainer.expected_value[1],
        shap_values[0],
        X_sample[idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.title(f"SHAP Local Explanation (Sample {idx})", fontsize=13)
    plt.tight_layout()
    save_dir = RESULTS_DIR if save_dir is None else save_dir
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / f"shap_local_sample_{idx}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Local explanation saved for sample {idx}")


def explain_feature_dependence(
    shap_values, X_data, feature_names, top_n: int = 4, save_dir: Path | None = None
):
    """
    Plot SHAP dependence plots for top features.
    Shows how a feature's value affects the prediction.
    """

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    _, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, idx in zip(axes.flat, top_indices, strict=False):
        feature_name = feature_names[idx]
        ax.scatter(
            X_data[:, idx],
            shap_values[:, idx],
            alpha=0.3,
            s=10,
            c=shap_values[:, idx],
            cmap="coolwarm",
        )
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel(f"SHAP value for {feature_name}", fontsize=11)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    plt.suptitle("SHAP Dependence Plots (Top Features)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_dir = RESULTS_DIR if save_dir is None else save_dir
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dependence plots saved to {save_dir / 'shap_dependence.png'}")


def run_explainability(model_path: str):
    """Run full SHAP explainability pipeline."""
    model = joblib.load(model_path)
    with np.load(RESULTS_DIR / "processed_data.npz") as archive:
        data = dict(archive.items())
    X_train, X_test = data["X_train"], data["X_test"]
    feature_names = joblib.load(RESULTS_DIR / "feature_names.pkl")

    print("=" * 50)
    print("MODEL EXPLAINABILITY (SHAP)")
    print("=" * 50)

    # Global explanations
    shap_values, explainer = explain_global(model, X_train, feature_names)

    # Local explanations for a few test samples
    for i in [0, 1, 2]:
        try:
            explain_local(model, explainer, X_test, feature_names, idx=i)
        except Exception as e:
            print(f"Could not generate local explanation for sample {i}: {e}")

    # Feature dependence
    explain_feature_dependence(shap_values, X_train[: len(shap_values)], feature_names)

    print("\nExplainability analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP model explainability")
    parser.add_argument(
        "--model_path", type=str, default=str(RESULTS_DIR / "best_model.pkl")
    )
    args = parser.parse_args()

    run_explainability(args.model_path)
