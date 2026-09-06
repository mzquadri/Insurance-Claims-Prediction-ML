"""Run a small preprocessing check without downloading external data."""

import sys
from pathlib import Path

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
    parts = encode_and_split(featured, validation_size=0.25)
    names = parts["feature_names"]

    for split in ("X_train", "X_validation", "X_test"):
        if parts[split] is None:
            raise SystemExit(f"{split} is missing; the three-way split did not run.")
        if parts[split].shape[1] != len(names):
            raise SystemExit(f"{split} does not match the documented feature names.")

    # The selection set has to be disjoint from both of the others, or the
    # threshold and calibrator choices have nowhere honest to be made.
    total = sum(len(parts[part]) for part in ("y_train", "y_validation", "y_test"))
    if total != len(featured):
        raise SystemExit("The three splits do not partition the rows.")

    print(f"Preprocessing smoke test passed: {parts['X_train'].shape[1]} features, "
          f"{len(parts['y_train'])} train / {len(parts['y_validation'])} validation / "
          f"{len(parts['y_test'])} test.")


if __name__ == "__main__":
    main()
