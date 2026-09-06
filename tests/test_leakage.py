"""Tests for the ways a score here can be flattering rather than wrong.

The failure this repository had was not a model trained on test data. It was two
scalars, a calibration method and a decision threshold, chosen by looking at the
test labels and then reported on those same labels. Every check that the fixed
path is clean is paired with one asserting the probe fires on the path that is
not, because a suite of only the first kind would pass on a probe that always
answered "no leak".
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src import data_pipeline
from src.benchmark import (
    best_threshold,
    business_value,
    score_at,
    selection_optimism,
)
from src.data_pipeline import clean_data, encode_and_split, engineer_features
from src.synthetic_portfolio import (
    NOT_FEATURES,
    bayes_limits,
    feature_columns,
    generate_portfolio,
)

POLICIES = 6_000


def portfolio(seed: int = 42):
    return generate_portfolio(n_policies=POLICIES, seed=seed)


def frame_for_pipeline(seed: int = 42):
    """The generated portfolio, shaped for the Kaggle-oriented pipeline."""
    frame = portfolio(seed).drop(columns=["true_probability"])
    return engineer_features(clean_data(frame))


class TheGeneratingProbabilityIsNotAFeature(unittest.TestCase):
    """The most direct leak available here, and the easiest to introduce."""

    def test_it_is_excluded_from_the_feature_list(self):
        frame = portfolio()
        self.assertNotIn("true_probability", feature_columns(frame))
        self.assertNotIn("is_claim", feature_columns(frame))

    def test_it_is_present_in_the_frame_to_begin_with(self):
        """Otherwise the exclusion would be doing nothing at all."""
        self.assertIn("true_probability", portfolio().columns)

    def test_the_exclusion_list_names_only_columns_that_exist(self):
        frame = portfolio()
        for name in NOT_FEATURES:
            with self.subTest(column=name):
                self.assertIn(name, frame.columns)

    def test_admitting_it_would_reach_the_floor_no_model_can_pass(self):
        """Why it has to stay out, measured rather than asserted.

        Not stated as a correlation. A Bernoulli draw correlates weakly with its
        own probability when claims are rare, so the correlation here is about
        0.16 and would look unremarkable. What matters is the score: handing the
        column over as a feature scores at the Bayes floor, which is the best
        result the data permits and is unreachable from the real features.
        """
        from sklearn.metrics import brier_score_loss

        frame = portfolio()
        floor = bayes_limits(frame["true_probability"], frame["is_claim"])["brier"]

        leaked = brier_score_loss(frame["is_claim"], frame["true_probability"])
        self.assertAlmostEqual(leaked, floor, places=12)

        base_rate = float(frame["is_claim"].mean())
        uninformed = float(((frame["is_claim"] - base_rate) ** 2).mean())
        self.assertLess(leaked, uninformed)

    def test_no_permitted_feature_gets_anywhere_near_that(self):
        """The paired check. If one did, the exclusion list is incomplete."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss

        frame = portfolio()
        floor = bayes_limits(frame["true_probability"], frame["is_claim"])["brier"]
        outcome = frame["is_claim"].to_numpy()
        base_rate = float(outcome.mean())
        uninformed = float(((outcome - base_rate) ** 2).mean())

        for column in feature_columns(frame):
            with self.subTest(feature=column):
                values = frame[[column]].to_numpy(dtype=float)
                model = LogisticRegression(max_iter=500).fit(values, outcome)
                score = brier_score_loss(outcome, model.predict_proba(values)[:, 1])
                # Comfortably short of the floor, and never below it.
                self.assertGreater(score, floor)
                self.assertGreater(score - floor, 0.3 * (uninformed - floor))


class ThreeWaySplit(unittest.TestCase):
    def setUp(self):
        self.frame = frame_for_pipeline()
        # Kept out of the working tree; see tests/test_selection_path.py.
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        with mock.patch.object(data_pipeline, "RESULTS_DIR",
                               Path(self._directory.name)):
            self.parts = encode_and_split(self.frame, validation_size=0.25)

    def test_all_three_parts_exist(self):
        for name in ("X_train", "X_validation", "X_test"):
            with self.subTest(part=name):
                self.assertIsNotNone(self.parts[name])

    def test_they_partition_the_rows(self):
        total = sum(len(self.parts[part])
                    for part in ("y_train", "y_validation", "y_test"))
        self.assertEqual(total, len(self.frame))

    def test_every_part_contains_both_classes(self):
        for name in ("y_train", "y_validation", "y_test"):
            with self.subTest(part=name):
                self.assertEqual(set(np.unique(self.parts[name])), {0, 1})

    def test_the_claim_rate_is_preserved_across_parts(self):
        rates = [self.parts[name].mean()
                 for name in ("y_train", "y_validation", "y_test")]
        self.assertLess(max(rates) - min(rates), 0.02)

    def test_asking_for_no_validation_still_returns_the_other_two(self):
        """The old two-way behaviour is still reachable, and says so."""
        with mock.patch.object(data_pipeline, "RESULTS_DIR",
                               Path(self._directory.name)):
            parts = encode_and_split(self.frame, validation_size=0.0)
        self.assertIsNone(parts["X_validation"])
        self.assertIsNone(parts["y_validation"])
        self.assertIsNotNone(parts["X_train"])

    def test_the_scaler_saw_only_training_rows(self):
        """Training columns are standardised; the others are not, by construction."""
        train_means = np.abs(self.parts["X_train"].mean(axis=0))
        self.assertLess(train_means.max(), 1e-9)


class ChoosingAThresholdOnTheSetItIsScoredOn(unittest.TestCase):
    """The defect this repository had, stated as a property."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.outcome = (rng.random(4_000) < 0.07).astype(int)
        self.probability = np.clip(
            0.07 + 0.05 * self.outcome + rng.normal(0, 0.04, 4_000), 0.001, 0.999)

    def test_it_can_never_score_worse_than_choosing_elsewhere(self):
        """Not a tendency. The search maximises the reported quantity."""
        for objective in ("f1", "business"):
            with self.subTest(objective=objective):
                result = selection_optimism(self.outcome, self.probability,
                                            objective, repeats=40)
                self.assertGreaterEqual(result["minimum_overstatement"], 0.0)

    def test_it_is_strictly_better_on_at_least_some_splits(self):
        """Otherwise the property above would be vacuous."""
        result = selection_optimism(self.outcome, self.probability, "f1", repeats=40)
        self.assertGreater(result["fraction_of_splits_overstated"], 0.0)

    def test_the_threshold_search_finds_the_maximum_it_is_shown(self):
        chosen = best_threshold(self.outcome, self.probability, "f1")
        at_chosen = score_at(self.outcome, self.probability, chosen, "f1")
        for other in (0.02, 0.05, 0.1, 0.3, 0.5, 0.9):
            with self.subTest(threshold=other):
                self.assertGreaterEqual(
                    at_chosen + 1e-12,
                    score_at(self.outcome, self.probability, other, "f1"))


class BusinessValue(unittest.TestCase):
    def test_flagging_nothing_pays_the_full_cost_of_every_claim(self):
        outcome = np.array([1, 0, 1, 0, 0])
        value = business_value(outcome, np.zeros(5, dtype=bool))
        self.assertEqual(value, 2 * -1000)

    def test_flagging_everything_pays_for_every_false_positive(self):
        outcome = np.array([1, 0, 1, 0, 0])
        value = business_value(outcome, np.ones(5, dtype=bool))
        self.assertEqual(value, 2 * 500 + 3 * -50)

    def test_a_perfect_call_is_the_best_available(self):
        outcome = np.array([1, 0, 1, 0, 0])
        perfect = business_value(outcome, outcome.astype(bool))
        for guess in (np.zeros(5, dtype=bool), np.ones(5, dtype=bool),
                      np.array([0, 1, 0, 1, 1], dtype=bool)):
            self.assertGreaterEqual(perfect, business_value(outcome, guess))


if __name__ == "__main__":
    unittest.main()
