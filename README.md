# Insurance Claims Prediction: Research Pipeline

Predicting whether a policy will file a claim is only half the problem — the model's probabilities also have to mean something, and someone has to explain them. This pipeline works through all three parts: binary classifiers (logistic regression, random forest, optional XGBoost/LightGBM), probability calibration with Platt scaling or isotonic regression, cost-sensitive threshold selection, and SHAP-based feature attribution. It is a learning and research artifact, not a production underwriting, pricing, fraud, or claims-decision system.

<p align="center">
  <img src="docs/diagrams/pipeline.svg" alt="Pipeline: Kaggle data to calibrated, explainable claim model" width="940">
</p>

## Scope and evidence

The repository contains source code and notebooks only. It does **not** version the Kaggle data, a train/test split, trained model, calibration output, or evaluation report. Consequently, this repository makes no verified accuracy, AUC, calibration-improvement, or business-value claim.

The pipeline can download the dataset named below through the Kaggle CLI, subject to Kaggle credentials, dataset availability, and the dataset's terms. Before using any external data, confirm that its license, intended use, privacy constraints, and feature definitions permit the proposed work. Do not use this code to make automated decisions about people or policies.

## Included methods

- Cleaning and optional feature engineering for common policy and vehicle columns.
- A train/test split with categorical encoders and numeric scaling fit on training data only.
- Logistic regression, random forest, and optional XGBoost/LightGBM comparisons.
- Cross-validated calibration and exploratory threshold analysis.
- Optional SHAP-based feature attribution.

The example cost matrix in `src/threshold_optimizer.py` is illustrative only. It is not a validated business policy and must not be used without domain, legal, fairness, and risk review.

## Data

The default download target is Kaggle's [Prediction of Insurance Claim](https://www.kaggle.com/datasets/thedevastator/prediction-of-insurance-claim). The expected target is `is_claim`; if a source uses another target name, pass a prepared dataset with that column or update the pipeline deliberately. `data/` and generated `results/` files are intentionally excluded from version control.

## Run locally

```bash
git clone https://github.com/mzquadri/Insurance-Claims-Prediction-ML.git
cd Insurance-Claims-Prediction-ML
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt

# Requires configured Kaggle credentials and acceptance of the source dataset terms.
python src/data_pipeline.py --download
python src/model_training.py --model random_forest --cv_folds 5
python src/calibration.py --model_path results/best_model.pkl
python src/threshold_optimizer.py --model_path results/calibrated_model.pkl
python src/explainability.py --model_path results/best_model.pkl
```

Calibration-method and decision-threshold selection should be evaluated on a validation set separate from the final untouched test set. The supplied scripts are exploratory and do not implement a complete governance or deployment workflow.

## Verify the checkout

```bash
python scripts/check_repository.py
python scripts/smoke_test.py
```

The smoke test uses a small in-memory fixture. It does not download data, train a claim model, or validate real-world performance.

## Project layout

```text
src/          Pipeline, modelling, calibration, threshold, and SHAP modules
notebooks/    Exploratory notebooks
scripts/      Repository and preprocessing smoke checks
results/      Ignored generated artifacts
docs/diagrams/  SVG overview of the pipeline
```

## License

MIT. See [LICENSE](LICENSE).
