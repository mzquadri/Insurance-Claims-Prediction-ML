# Insurance Claims Prediction - ML Pipeline

End-to-end machine learning solution for insurance claims prediction, featuring model calibration, threshold optimization, and business-aligned performance metrics. This project demonstrates a production-grade approach to building ML models for insurance risk assessment.

## Project Overview

Predicting insurance claim likelihood and severity is critical for pricing, underwriting, and fraud detection. This project builds a complete ML pipeline that:

- **Feature engineering**: Domain-specific feature construction from policyholder and claim data
- **Model selection**: Comparison of Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Calibration**: Platt scaling and isotonic regression for reliable probability estimates
- **Thresholding**: Business-driven threshold optimization balancing precision/recall trade-offs
- **Explainability**: SHAP values for model interpretability and feature importance

## Dataset

**Insurance Claims Dataset** from [Kaggle](https://www.kaggle.com/datasets/thedevastator/prediction-of-insurance-claim)
- Features: policyholder demographics, vehicle info, policy details
- Target: Binary classification (claim filed vs. no claim)

## Project Structure

```
Insurance-Claims-Prediction-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      # Data loading, cleaning, feature engineering
│   ├── model_training.py     # Model training with cross-validation
│   ├── calibration.py        # Probability calibration methods
│   ├── threshold_optimizer.py # Business-driven threshold selection
│   └── explainability.py     # SHAP-based model explanations
├── notebooks/
│   ├── 01_EDA_and_Feature_Engineering.ipynb
│   └── 02_Model_Training_and_Evaluation.ipynb
├── data/                     # Dataset directory
└── results/                  # Evaluation plots and metrics
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mzquadri/Insurance-Claims-Prediction-ML.git
cd Insurance-Claims-Prediction-ML

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/data_pipeline.py --download
python src/model_training.py --model xgboost --cv_folds 5
python src/calibration.py --model_path results/best_model.pkl
python src/threshold_optimizer.py --model_path results/calibrated_model.pkl
```

## Results

| Model | Accuracy | AUC-ROC | Brier Score | Log Loss |
|-------|----------|---------|-------------|----------|
| Logistic Regression | 82.3% | 0.841 | 0.142 | 0.421 |
| Random Forest | 85.7% | 0.893 | 0.118 | 0.378 |
| XGBoost | 87.2% | 0.912 | 0.104 | 0.341 |
| LightGBM | 87.5% | 0.916 | 0.101 | 0.335 |

### Calibration Results

Post-calibration Brier scores improve by ~15%, yielding more reliable probability estimates for downstream business decisions.

## Key Features

- **Probability Calibration**: Ensures predicted probabilities align with true event frequencies
- **Threshold Optimization**: Selects decision threshold that maximizes business value (cost-sensitive)
- **SHAP Explainability**: Global and local feature importance for transparent decision-making
- **Cross-Validation**: Stratified K-Fold with proper data leakage prevention

## Technical Stack

- **ML**: scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP
- **Visualization**: Matplotlib, Seaborn
- **Data**: Pandas, NumPy

## Author

**Mohd Zamin Quadri** - M.Sc. Mathematics in Science and Engineering, Technical University of Munich

[![LinkedIn](https://img.shields.io/badge/LinkedIn-mohd--zamin-blue)](https://www.linkedin.com/in/mohd-zamin/)
[![GitHub](https://img.shields.io/badge/GitHub-mzquadri-black)](https://github.com/mzquadri)
