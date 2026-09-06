"""
Data pipeline for insurance claims prediction.
Handles data loading, cleaning, feature engineering, and train/test splitting.
"""

import argparse
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

KAGGLE_DATASET = "thedevastator/prediction-of-insurance-claim"


def download_dataset(data_dir: Path = DATA_DIR):
    """Download insurance claims dataset from Kaggle."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if list(data_dir.glob("*.csv")):
        print("Dataset already exists.")
        return
    print("Downloading insurance claims dataset...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(data_dir),
            "--unzip",
        ],
        check=True,
    )
    print(f"Downloaded to {data_dir}")


def load_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and combine raw data files."""
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Run with --download first."
        )
    df = pd.read_csv(csv_files[0])
    print(f"Loaded {len(df)} records with {df.shape[1]} features")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data: handle missing values, fix types, remove duplicates."""
    df = df.copy()

    # Drop ID-like columns if present
    id_cols = [
        c for c in df.columns if "id" in c.lower() and df[c].nunique() == len(df)
    ]
    df.drop(columns=id_cols, inplace=True, errors="ignore")

    # Drop duplicates
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {n_before - len(df)} duplicates")

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific features for insurance prediction."""
    df = df.copy()

    # Interaction features (if relevant columns exist)
    if "age" in df.columns and "annual_premium" in df.columns:
        df["premium_per_age"] = df["annual_premium"] / (df["age"] + 1)

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )

    if "vehicle_age" in df.columns:
        vehicle_age_map = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
        if df["vehicle_age"].dtype == "object":
            df["vehicle_age_numeric"] = df["vehicle_age"].map(vehicle_age_map).fillna(1)

    if "annual_premium" in df.columns:
        df["log_premium"] = np.log1p(df["annual_premium"])
        df["premium_quartile"] = pd.qcut(
            df["annual_premium"], q=4, labels=False, duplicates="drop"
        )

    print(f"Features after engineering: {df.shape[1]}")
    return df


def encode_and_split(
    df: pd.DataFrame,
    target_col: str = "is_claim",
    test_size: float = 0.2,
    validation_size: float = 0.25,
    random_state: int = 42,
):
    """
    Encode categorical variables, scale numerics, and split into three parts.

    `validation_size` is a fraction of the training rows, not of the whole frame.
    Pass 0 to get only train and test, in which case "X_validation" and
    "y_validation" come back as None and any label-driven selection has nowhere
    honest to happen.

    Returns
    -------
    dict with X_train, y_train, X_validation, y_validation, X_test, y_test and
    feature_names. This used to be a five-tuple of train and test only; the names
    are explicit because there are now three parts and mixing them up is exactly
    the failure this split exists to prevent.
    """
    # Identify target
    target_candidates = [
        c for c in df.columns if "claim" in c.lower() or "target" in c.lower()
    ]
    if target_col not in df.columns and target_candidates:
        target_col = target_candidates[0]
        print(f"Using '{target_col}' as target column")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found. Available columns: "
            f"{', '.join(df.columns)}"
        )

    y = df[target_col].values
    X = df.drop(columns=[target_col]).copy()

    # Split before fitting preprocessing artifacts so test-set categories and
    # distributions cannot influence the training representation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # A third part, carved out of training, for decisions that have to be made by
    # looking at labels: which calibrator to keep, and where to put the decision
    # threshold. Without it there is nowhere to make those choices except the test
    # set, and a choice made there cannot also be scored there. See src/benchmark.py.
    X_validation, y_validation = None, None
    if validation_size > 0:
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train, y_train, test_size=validation_size,
            random_state=random_state, stratify=y_train,
        )

    # Encode categoricals
    label_encoders = {}
    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        unknown_value = -1
        mapping = {label: index for index, label in enumerate(le.classes_)}
        X_test[col] = X_test[col].astype(str).map(mapping).fillna(unknown_value).astype(int)
        if X_validation is not None:
            X_validation[col] = (X_validation[col].astype(str).map(mapping)
                                 .fillna(unknown_value).astype(int))
        label_encoders[col] = le

    feature_names = X_train.columns.tolist()

    # Scale numeric features. Fitted on training rows only, so neither the
    # validation nor the test set influences the representation.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if X_validation is not None:
        X_validation = scaler.transform(X_validation)

    # Save preprocessor artifacts
    RESULTS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, RESULTS_DIR / "scaler.pkl")
    joblib.dump(label_encoders, RESULTS_DIR / "label_encoders.pkl")

    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples", end="")
    print(f" | Validation: {X_validation.shape[0]} samples"
          if X_validation is not None else " | no validation split requested")
    print(f"Positive rate: {y_train.mean():.3f} (train), {y_test.mean():.3f} (test)")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_validation": X_validation, "y_validation": y_validation,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names,
    }


def run_pipeline(data_dir: Path = DATA_DIR):
    """Execute the full data pipeline."""
    df = load_data(data_dir)
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, feature_names = encode_and_split(df)

    # Save processed data
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(
        RESULTS_DIR / "processed_data.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    joblib.dump(feature_names, RESULTS_DIR / "feature_names.pkl")
    print(f"\nProcessed data saved to {RESULTS_DIR}")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insurance claims data pipeline")
    parser.add_argument(
        "--download", action="store_true", help="Download dataset from Kaggle"
    )
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    if args.download:
        download_dataset(Path(args.data_dir))

    run_pipeline(Path(args.data_dir))
