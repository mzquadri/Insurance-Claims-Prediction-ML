"""Tests for the hand-off between the script that writes and the ones that read.

`src/data_pipeline.py` writes `results/processed_data.npz`, and four other
modules load it and nothing else. The existing suite exercised both ends and
neither of them together: `tests/test_selection_path.py` builds that archive
itself, with `np.savez`, so it proves the consumers work on an archive of the
right shape without ever asking whether the producer writes one. It did not.
`run_pipeline` unpacked a dict of seven entries into five names and raised before
writing anything, and even past that it saved four of the six arrays, so the
validation split that `run_calibration` and `run_threshold_optimization` insist
on was absent. Both of them tell the reader to rerun the script that could not
run.

So these tests drive the real producer and hand its output to the real
consumers. Nothing here constructs the archive; if a test wants one, the shipped
code path has to make it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from src import calibration, data_pipeline, model_training, threshold_optimizer
from src.synthetic_portfolio import generate_portfolio

POLICIES = 4_000


def csv_directory(directory: Path) -> Path:
    """A CSV where the Kaggle path expects one, shaped like the real target."""
    data_dir = directory / "data"
    data_dir.mkdir()
    frame = generate_portfolio(n_policies=POLICIES, seed=7).drop(
        columns=["true_probability"])
    frame.to_csv(data_dir / "claims.csv", index=False)
    return data_dir


class TheArchiveThePipelineWrites(unittest.TestCase):
    """`python -m src.data_pipeline` has to produce something usable."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)
        self.data_dir = csv_directory(self.directory)
        self._patch = mock.patch.object(
            data_pipeline, "RESULTS_DIR", self.directory)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_it_runs_at_all(self):
        """It used to raise ValueError before writing a byte."""
        parts = data_pipeline.run_pipeline(self.data_dir)
        self.assertIsInstance(parts, dict)
        self.assertTrue((self.directory / "processed_data.npz").is_file())

    def test_it_writes_every_part_the_split_produced(self):
        data_pipeline.run_pipeline(self.data_dir)
        with np.load(self.directory / "processed_data.npz") as archive:
            written = set(archive.files)
        self.assertEqual(
            written,
            {"X_train", "y_train", "X_validation", "y_validation",
             "X_test", "y_test"})

    def test_the_three_parts_partition_the_rows(self):
        data_pipeline.run_pipeline(self.data_dir)
        with np.load(self.directory / "processed_data.npz") as archive:
            total = sum(len(archive[name])
                        for name in ("y_train", "y_validation", "y_test"))
        self.assertEqual(total, POLICIES)

    def test_the_feature_names_match_the_matrix_it_wrote(self):
        data_pipeline.run_pipeline(self.data_dir)
        names = joblib.load(self.directory / "feature_names.pkl")
        with np.load(self.directory / "processed_data.npz") as archive:
            for part in ("X_train", "X_validation", "X_test"):
                with self.subTest(part=part):
                    self.assertEqual(archive[part].shape[1], len(names))


class WhatTheConsumersDoWithIt(unittest.TestCase):
    """The producer and the consumers, joined, with nothing in between."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)
        data_dir = csv_directory(self.directory)

        for module in (data_pipeline, calibration, threshold_optimizer):
            patch = mock.patch.object(module, "RESULTS_DIR", self.directory)
            patch.start()
            self.addCleanup(patch.stop)

        parts = data_pipeline.run_pipeline(data_dir)
        model = LogisticRegression(max_iter=1000).fit(
            parts["X_train"], parts["y_train"])
        self.model_path = self.directory / "best_model.pkl"
        joblib.dump(model, self.model_path)

    def test_calibration_accepts_it(self):
        best = calibration.run_calibration(str(self.model_path))
        self.assertTrue((self.directory / "calibrated_model.pkl").is_file())
        with np.load(self.directory / "processed_data.npz") as archive:
            probability = best.predict_proba(archive["X_test"])[:, 1]
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))

    def test_threshold_optimisation_accepts_it(self):
        thresholds = threshold_optimizer.run_threshold_optimization(
            str(self.model_path))
        self.assertEqual(set(thresholds), {"f1", "youden", "business"})
        for name, value in thresholds.items():
            with self.subTest(method=name):
                self.assertGreater(value, 0.0)
                self.assertLess(value, 1.0)

    def test_the_thresholds_were_chosen_on_the_validation_rows(self):
        """Which set the search saw, read off the answer rather than the code.

        A threshold chosen on 8,000 rows of a 0.01 grid lands where those rows
        put it. Recomputing the search on the validation rows has to land in the
        same place, and recomputing it on the test rows generally does not.
        """
        thresholds = threshold_optimizer.run_threshold_optimization(
            str(self.model_path))
        model = joblib.load(self.model_path)
        with np.load(self.directory / "processed_data.npz") as archive:
            data = dict(archive.items())

        on_validation, _, _ = threshold_optimizer.find_optimal_threshold(
            data["y_validation"],
            model.predict_proba(data["X_validation"])[:, 1], method="f1")
        self.assertAlmostEqual(thresholds["f1"], on_validation, places=10)


class TheCrossValidationComparison(unittest.TestCase):
    """`cross_validate_models` reported `Brier Score: nan` and said nothing.

    scikit-learn removed `needs_proba` in 1.6 and now forwards an unrecognised
    keyword to the metric, so the scorer raised on every fold, cross_validate
    replaced each score with nan, and the module's blanket warnings filter
    swallowed the explanation.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        cls.X = rng.normal(size=(1_200, 4))
        cls.y = (rng.random(1_200) < 0.2).astype(int)
        cls.models = {"logistic_regression": LogisticRegression(max_iter=500)}
        cls.results = model_training.cross_validate_models(
            cls.X, cls.y, models=cls.models, n_folds=3)

    def test_every_reported_metric_is_a_number(self):
        for metric, value in self.results["logistic_regression"].items():
            with self.subTest(metric=metric):
                self.assertFalse(np.isnan(value), f"{metric} came back nan")

    def test_the_brier_score_is_in_range(self):
        """nan passed a range check written with `<`, so this is explicit."""
        brier = self.results["logistic_regression"]["brier_score"]
        self.assertTrue(0.0 <= brier <= 1.0, brier)

    def test_it_matches_the_same_cross_validation_done_by_hand(self):
        """The scorer, checked against the quantity it is supposed to compute.

        Not a copy of the scorer: this fits the folds itself and calls
        brier_score_loss on the positive-class column directly. A scorer built
        on the wrong response method scores hard labels and disagrees here.
        """
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train, test in folds.split(self.X, self.y):
            model = LogisticRegression(max_iter=500).fit(
                self.X[train], self.y[train])
            scores.append(brier_score_loss(
                self.y[test], model.predict_proba(self.X[test])[:, 1]))

        self.assertAlmostEqual(
            self.results["logistic_regression"]["brier_score"],
            float(np.mean(scores)), places=10)

    def test_scoring_hard_labels_instead_would_look_different(self):
        """Otherwise the check above would pass on the wrong response method."""
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train, test in folds.split(self.X, self.y):
            model = LogisticRegression(max_iter=500).fit(
                self.X[train], self.y[train])
            scores.append(brier_score_loss(
                self.y[test], model.predict(self.X[test])))
        self.assertNotAlmostEqual(
            self.results["logistic_regression"]["brier_score"],
            float(np.mean(scores)), places=4)


if __name__ == "__main__":
    unittest.main()
