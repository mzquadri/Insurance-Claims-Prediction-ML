"""Run a small preprocessing check without downloading external data."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import clean_data, encode_and_split, engineer_features


def main() -> None:
    rows = 20
    data = pd.DataFrame(
        {
            "age": list(range(21, 21 + rows)),
            "annual_premium": [1000 + 10 * index for index in range(rows)],
            "vehicle_age": ["< 1 Year", "1-2 Year"] * (rows // 2),
            "region": ["north", "south"] * (rows // 2),
            "is_claim": [0, 1] * (rows // 2),
        }
    )
    cleaned = clean_data(data)
    featured = engineer_features(cleaned)
    X_train, X_test, y_train, y_test, feature_names = encode_and_split(featured)

    if X_train.shape[1] != len(feature_names) or X_test.shape[1] != len(feature_names):
        raise SystemExit("Feature matrices do not match the documented feature names.")
    if set(y_train) != {0, 1} or set(y_test) != {0, 1}:
        raise SystemExit("Stratified split did not preserve both classes.")

    print(f"Preprocessing smoke test passed: {X_train.shape[1]} features.")


if __name__ == "__main__":
    main()
