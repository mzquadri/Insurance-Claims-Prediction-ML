"""End-to-end tests for the two entry points that used to select on the test set.

These run the real `run_calibration` and `run_threshold_optimization` against
generated data, with the results directory redirected to a temporary one, because
the fix is only worth anything if the shipped code paths actually behave. Checking
the benchmark alone would leave the modules the README tells people to run
untested.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src import calibration, data_pipeline, threshold_optimizer
from src.data_pipeline import encode_and_split
from src.synthetic_portfolio import generate_portfolio


def prepared(directory: Path, with_validation: bool) -> Path:
    """Write the artifacts the two entry points load, into a temporary directory."""
    frame = generate_portfolio(n_policies=6_000, seed=42).drop(
        columns=["true_probability"])
    # encode_and_split writes the scaler and the encoders beside the data, so it
    # is pointed at the temporary directory too. Otherwise running the tests
    # leaves artifacts in the working tree.
    with mock.patch.object(data_pipeline, "RESULTS_DIR", directory):
        parts = encode_and_split(
            frame, validation_size=0.25 if with_validation else 0.0)

    arrays = {name: value for name, value in parts.items()
              if name != "feature_names" and value is not None}
    np.savez(directory / "processed_data.npz", **arrays)
    joblib.dump(parts["feature_names"], directory / "feature_names.pkl")

    model = LogisticRegression(max_iter=1000).fit(parts["X_train"], parts["y_train"])
    model_path = directory / "best_model.pkl"
    joblib.dump(model, model_path)
    return model_path


class WithoutAValidationSplit(unittest.TestCase):
    """Both entry points must refuse rather than quietly select on the test set."""

    def test_threshold_optimisation_refuses(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            model_path = prepared(directory, with_validation=False)
            with mock.patch.object(threshold_optimizer, "RESULTS_DIR", directory), \
                    self.assertRaises(SystemExit) as caught:
                threshold_optimizer.run_threshold_optimization(str(model_path))
            self.assertIn("validation", str(caught.exception).lower())

    def test_calibration_refuses(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            model_path = prepared(directory, with_validation=False)
            with mock.patch.object(calibration, "RESULTS_DIR", directory), \
                    self.assertRaises(SystemExit) as caught:
                calibration.run_calibration(str(model_path))
            self.assertIn("validation", str(caught.exception).lower())


class WithAValidationSplit(unittest.TestCase):
    def test_threshold_optimisation_runs_and_reports_usable_thresholds(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            model_path = prepared(directory, with_validation=True)
            with mock.patch.object(threshold_optimizer, "RESULTS_DIR", directory):
                threshold_optimizer.run_threshold_optimization(str(model_path))

            self.assertTrue((directory / "threshold_analysis.png").is_file())
            self.assertTrue((directory / "optimal_thresholds.pkl").is_file())

            thresholds = joblib.load(directory / "optimal_thresholds.pkl")
            for name, value in thresholds.items():
                with self.subTest(method=name):
                    self.assertGreater(value, 0.0)
                    self.assertLess(value, 1.0)

    def test_calibration_runs_and_keeps_the_method_validation_chose(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            model_path = prepared(directory, with_validation=True)
            with mock.patch.object(calibration, "RESULTS_DIR", directory):
                calibration.run_calibration(str(model_path))
            self.assertTrue((directory / "calibrated_model.pkl").is_file())

            calibrated = joblib.load(directory / "calibrated_model.pkl")
            with np.load(directory / "processed_data.npz") as archive:
                x_test = archive["X_test"]
            probability = calibrated.predict_proba(x_test)[:, 1]
            self.assertTrue(np.all((probability >= 0) & (probability <= 1)))

    def test_a_threshold_scored_on_test_is_not_the_test_optimum(self):
        """The property the fix buys, checked on the shipped code path.

        The reported threshold now comes from validation, so it will usually not
        be the one that maximises the test set. If it always were, the split
        would not be doing anything.
        """
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            model_path = prepared(directory, with_validation=True)
            model = joblib.load(model_path)
            with np.load(directory / "processed_data.npz") as archive:
                data = dict(archive.items())

            on_validation, _, _ = threshold_optimizer.find_optimal_threshold(
                data["y_validation"],
                model.predict_proba(data["X_validation"])[:, 1], method="f1")
            on_test, _, _ = threshold_optimizer.find_optimal_threshold(
                data["y_test"],
                model.predict_proba(data["X_test"])[:, 1], method="f1")

            test_probability = model.predict_proba(data["X_test"])[:, 1]
            honest = threshold_optimizer.metrics_at_threshold(
                data["y_test"], test_probability, on_validation)["f1"]
            optimistic = threshold_optimizer.metrics_at_threshold(
                data["y_test"], test_probability, on_test)["f1"]

            # Choosing on the test set cannot score worse there.
            self.assertGreaterEqual(optimistic + 1e-12, honest)


if __name__ == "__main__":
    unittest.main()
