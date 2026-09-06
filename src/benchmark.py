"""Measure what choosing a threshold and a calibrator on the test set costs.

    python -m src.benchmark

This repository's README already states the correct procedure: "Calibration-method
and decision-threshold selection should be evaluated on a validation set separate
from the final untouched test set." The code did the opposite. `run_calibration`
compared three calibrators by their Brier score on the test set and kept the
winner, and `run_threshold_optimization` searched a hundred thresholds against the
test labels and then reported the metrics at the threshold it had just chosen. No
validation split existed to do otherwise with.

Fitting a model on the test set is the leak everyone checks for. Selecting a
scalar on it is the same mistake wearing a smaller hat, and it survives because
the model itself was trained correctly, so every audit of the training path comes
back clean.

The data is generated, so the true claim probability behind every outcome is
known. That permits two things real data does not. Calibration can be scored
against the probability it is meant to estimate rather than against binned
frequencies, and the best attainable score is computable, so a Brier score can be
read against the floor instead of against nothing.

Writes results/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .synthetic_portfolio import (
    SEED,
    bayes_limits,
    feature_columns,
    generate_portfolio,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark.json"

N_POLICIES = 40_000

#: Illustrative only, carried over from src/threshold_optimizer.py so the two
#: agree. These are not a validated business policy and the asymmetry between
#: them, twenty to one, is what makes the default threshold of 0.5 wrong.
COSTS = {"tp_value": 500, "tn_value": 0, "fp_cost": -50, "fn_cost": -1000}

THRESHOLDS = np.arange(0.01, 1.00, 0.01)


def business_value(outcome, predicted, costs=COSTS) -> float:
    """Value of a set of decisions under the illustrative cost matrix."""
    outcome = np.asarray(outcome).astype(bool)
    predicted = np.asarray(predicted).astype(bool)
    return float(
        np.sum(outcome & predicted) * costs["tp_value"]
        + np.sum(~outcome & ~predicted) * costs["tn_value"]
        + np.sum(~outcome & predicted) * costs["fp_cost"]
        + np.sum(outcome & ~predicted) * costs["fn_cost"])


def score_at(outcome, probability, threshold: float, objective: str) -> float:
    """Score the decisions a threshold produces, under the chosen objective."""
    predicted = np.asarray(probability) >= threshold
    if objective == "f1":
        return float(f1_score(outcome, predicted, zero_division=0))
    if objective == "business":
        return business_value(outcome, predicted)
    raise ValueError(objective)


def best_threshold(outcome, probability, objective: str) -> float:
    """Search thresholds against whatever labels it is handed.

    Which labels those are is the entire question this file exists to answer.
    """
    scores = [score_at(outcome, probability, threshold, objective)
              for threshold in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def selection_optimism(outcome, probability, objective: str, repeats: int = 200,
                       seed: int = SEED) -> dict:
    """How much a threshold chosen on the scoring set flatters itself.

    The model is held fixed and only the partition changes, because the model is
    not what is being tested. Each repeat splits the held-out rows in half, picks
    a threshold on one half, and scores it on the other, against picking and
    scoring on that same other half.

    One split cannot answer this. The optimism is a variance effect: the chosen
    threshold absorbs whichever way the noise in that particular set happens to
    fall, so on any single split it can be large, small, or absent. What is stable
    is its average over repartitions, and that it is never negative in expectation.
    """
    outcome = np.asarray(outcome)
    probability = np.asarray(probability)
    rng = np.random.default_rng(seed)
    half = len(outcome) // 2

    optimistic, honest, gaps = [], [], []
    for _ in range(repeats):
        order = rng.permutation(len(outcome))
        pick, score = order[:half], order[half:]

        scored_outcome, scored_probability = outcome[score], probability[score]
        on_score = best_threshold(scored_outcome, scored_probability, objective)
        on_pick = best_threshold(outcome[pick], probability[pick], objective)

        optimistic.append(
            score_at(scored_outcome, scored_probability, on_score, objective))
        honest.append(
            score_at(scored_outcome, scored_probability, on_pick, objective))
        gaps.append(optimistic[-1] - honest[-1])

    gaps = np.asarray(gaps, dtype=float)
    return {
        "repeats": repeats,
        "mean_reported_when_chosen_on_the_scoring_set": float(np.mean(optimistic)),
        "mean_reported_when_chosen_on_a_separate_set": float(np.mean(honest)),
        "mean_overstatement": float(gaps.mean()),
        "median_overstatement": float(np.median(gaps)),
        # The gap cannot be negative. The threshold chosen on the scoring set is
        # by definition the one that maximises the objective there, so it can
        # only tie or beat a threshold chosen anywhere else. That is the whole
        # argument: the number is not an unbiased estimate that happened to come
        # out high, it is an upper bound reported as if it were an estimate.
        "minimum_overstatement": float(gaps.min()),
        "fraction_of_splits_overstated": float((gaps > 0).mean()),
        "fraction_of_splits_where_both_choices_tie": float((gaps == 0).mean()),
        "worst_single_split_overstatement": float(gaps.max()),
    }


def expected_calibration_error(outcome, probability, bins: int = 20) -> float:
    """The usual bin-based measure, for comparison with the exact one below."""
    outcome = np.asarray(outcome, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.quantile(probability, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    total = 0.0
    for low, high in pairwise(edges):
        mask = (probability > low) & (probability <= high)
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(outcome[mask].mean() - probability[mask].mean())
    return float(total)


def probability_scores(outcome, probability, truth) -> dict:
    """Score a set of predicted probabilities, including against the truth.

    `mean_absolute_error_against_truth` is the one a real dataset cannot provide.
    Everything else here is computable without knowing the generating process,
    and is reported alongside so the two can be compared.
    """
    return {
        "brier": float(brier_score_loss(outcome, probability)),
        "log_loss": float(log_loss(outcome, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(outcome, probability)),
        "expected_calibration_error": expected_calibration_error(outcome, probability),
        "mean_absolute_error_against_truth": float(
            np.abs(np.asarray(probability) - np.asarray(truth)).mean()),
        "mean_predicted_probability": float(np.mean(probability)),
    }


def build_model(name: str):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=SEED)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                      random_state=SEED, n_jobs=-1)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=250, random_state=SEED)
    raise ValueError(name)


def main(n_policies: int = N_POLICIES, out: Path = OUT) -> int:
    frame = generate_portfolio(n_policies=n_policies, seed=SEED)
    columns = feature_columns(frame)

    # Three-way split. The repository had only train and test, which left nowhere
    # to make a selection decision except on the test set.
    train, holdout = train_test_split(frame, test_size=0.40, random_state=SEED,
                                      stratify=frame["is_claim"])
    validation, test = train_test_split(holdout, test_size=0.50, random_state=SEED,
                                        stratify=holdout["is_claim"])

    scaler = StandardScaler().fit(train[columns])
    sets = {name: (scaler.transform(part[columns]),
                   part["is_claim"].to_numpy(),
                   part["true_probability"].to_numpy())
            for name, part in (("train", train), ("validation", validation),
                               ("test", test))}

    print(f"  {len(frame):,} generated policies, {len(columns)} features")
    print(f"  claim rate {frame['is_claim'].mean() * 100:.2f} percent")
    print(f"  train {len(train):,}   validation {len(validation):,}   "
          f"test {len(test):,}\n")

    floor = bayes_limits(sets["test"][2], sets["test"][1])
    test_outcome = sets["test"][1]
    reference_decisions = {
        "never_flag_a_policy": business_value(test_outcome,
                                              np.zeros(len(test_outcome), dtype=bool)),
        "flag_every_policy": business_value(test_outcome,
                                            np.ones(len(test_outcome), dtype=bool)),
        "oracle_rule_on_the_true_probability": business_value(
            test_outcome,
            sets["test"][2] >= best_threshold(test_outcome, sets["test"][2], "business")),
    }
    base_rate = float(sets["train"][1].mean())
    never_claim_accuracy = float((sets["test"][1] == 0).mean())

    models, calibration, thresholds = {}, {}, {}

    for name in ("logistic_regression", "random_forest", "gradient_boosting"):
        model = build_model(name).fit(sets["train"][0], sets["train"][1])
        raw = {part: model.predict_proba(sets[part][0])[:, 1] for part in sets}

        models[name] = {
            "test": probability_scores(sets["test"][1], raw["test"], sets["test"][2]),
            "validation": probability_scores(sets["validation"][1], raw["validation"],
                                             sets["validation"][2]),
        }
        print(f"  {name:<20} test Brier {models[name]['test']['brier']:.5f}   "
              f"AUC {models[name]['test']['roc_auc']:.4f}   "
              f"ECE {models[name]['test']['expected_calibration_error']:.4f}")

        # Calibrators are fitted on the training rows only, with internal folds.
        # The question is which of them gets picked, and on what.
        variants = {"uncalibrated": raw}
        for method in ("sigmoid", "isotonic"):
            calibrated = CalibratedClassifierCV(
                build_model(name), method=method, cv=5)
            calibrated.fit(sets["train"][0], sets["train"][1])
            variants[method] = {part: calibrated.predict_proba(sets[part][0])[:, 1]
                                for part in sets}

            # The same calibrator applied as a pure post-hoc map: the fitted model
            # is frozen and only the one-dimensional correction is learned, on the
            # validation rows. The distinction matters for what can be claimed.
            # With cv=5 the call above trains five fresh models and averages them,
            # so it changes the ranking as well as the probabilities. Only this
            # form leaves the ordering the base model produced.
            # FrozenEstimator replaces the cv="prefit" argument, which scikit-learn
            # removed in 1.8. It wraps the fitted model so the calibrator cannot
            # refit it, which is the whole point of the comparison.
            posthoc = CalibratedClassifierCV(
                FrozenEstimator(model), method=method)
            posthoc.fit(sets["validation"][0], sets["validation"][1])
            variants[f"posthoc_{method}"] = {
                part: posthoc.predict_proba(sets[part][0])[:, 1] for part in sets}

        scored = {
            method: {
                "validation": probability_scores(sets["validation"][1],
                                                 values["validation"],
                                                 sets["validation"][2]),
                "test": probability_scores(sets["test"][1], values["test"],
                                           sets["test"][2]),
            }
            for method, values in variants.items()
        }

        candidates = ("sigmoid", "isotonic")
        chosen_on_test = min(candidates, key=lambda m: scored[m]["test"]["brier"])
        chosen_on_validation = min(
            candidates, key=lambda m: scored[m]["validation"]["brier"])

        calibration[name] = {
            "methods": scored,
            "chosen_on_test": chosen_on_test,
            "chosen_on_validation": chosen_on_validation,
            "selection_agrees": chosen_on_test == chosen_on_validation,
            "reported_brier_when_chosen_on_test":
                scored[chosen_on_test]["test"]["brier"],
            "reported_brier_when_chosen_on_validation":
                scored[chosen_on_validation]["test"]["brier"],
            "auc_by_method": {
                method: scored[method]["test"]["roc_auc"] for method in scored},
            # Platt is strictly monotone, so applied post hoc it cannot reorder
            # anything and the AUC must be identical, not merely close. Recorded
            # as a difference so the claim is checkable rather than asserted.
            "posthoc_platt_auc_shift": abs(
                scored["posthoc_sigmoid"]["test"]["roc_auc"]
                - scored["uncalibrated"]["test"]["roc_auc"]),
            "refit_platt_auc_shift": abs(
                scored["sigmoid"]["test"]["roc_auc"]
                - scored["uncalibrated"]["test"]["roc_auc"]),
        }

        # The threshold, chosen two ways, both reported on the test set.
        best = variants[chosen_on_validation]
        entry = {}
        for objective in ("f1", "business"):
            on_test = best_threshold(sets["test"][1], best["test"], objective)
            on_validation = best_threshold(sets["validation"][1], best["validation"],
                                           objective)

            entry[objective] = {
                "threshold_chosen_on_test": on_test,
                "threshold_chosen_on_validation": on_validation,
                "reported_when_chosen_on_test": score_at(
                    test_outcome, best["test"], on_test, objective),
                "reported_when_chosen_on_validation": score_at(
                    test_outcome, best["test"], on_validation, objective),
                "at_the_default_half": score_at(
                    test_outcome, best["test"], 0.5, objective),
            }
            optimistic = entry[objective]["reported_when_chosen_on_test"]
            honest = entry[objective]["reported_when_chosen_on_validation"]
            entry[objective]["overstated_by"] = float(optimistic - honest)
            entry[objective]["across_repartitions"] = selection_optimism(
                np.concatenate([sets["validation"][1], test_outcome]),
                np.concatenate([best["validation"], best["test"]]),
                objective)
        thresholds[name] = entry

    payload = {
        "environment": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "data": {
            "source": "generated by src.synthetic_portfolio",
            "is_synthetic": True,
            "seed": SEED,
            "policies": len(frame),
            "features": len(columns),
            "claim_rate": float(frame["is_claim"].mean()),
            "train": len(train), "validation": len(validation), "test": len(test),
            "warning": "no insurance data was used; the relationships were written "
                       "in src/synthetic_portfolio.py and carry no actuarial meaning",
        },
        "reference_points": {
            "bayes_floor": floor,
            "business_value_of_fixed_rules": reference_decisions,
            "train_base_rate": base_rate,
            "accuracy_by_never_predicting_a_claim": never_claim_accuracy,
            "brier_predicting_the_base_rate": float(
                ((sets["test"][1] - base_rate) ** 2).mean()),
        },
        "costs": COSTS,
        "models": models,
        "calibration": calibration,
        "thresholds": thresholds,
    }

    print(f"\n  a model that knew every true probability would score "
          f"{floor['brier']:.5f}")
    print(f"  predicting the base rate for every policy scores "
          f"{payload['reference_points']['brier_predicting_the_base_rate']:.5f}")
    print(f"  never predicting a claim is {never_claim_accuracy * 100:.2f} percent "
          f"accurate\n")

    print("  choosing the calibrator on the test set against on validation:")
    for name, entry in calibration.items():
        print(f"    {name:<20} test picks {entry['chosen_on_test']:<9} "
              f"validation picks {entry['chosen_on_validation']:<9} "
              f"{'same' if entry['selection_agrees'] else 'DIFFERENT'}")

    print("\n  business value of the rules a threshold competes with:")
    for label, value in reference_decisions.items():
        print(f"    {label:<38} {value:>12,.0f}")

    print("\n  a threshold chosen on the set it is then scored on, "
          f"over {thresholds['logistic_regression']['f1']['across_repartitions']['repeats']} "
          "repartitions:")
    for name, entry in thresholds.items():
        for objective, values in entry.items():
            across = values["across_repartitions"]
            fmt = "{:>11,.0f}" if objective == "business" else "{:>11.4f}"
            print(f"    {name:<20} {objective:<9} "
                  f"chosen on it "
                  f"{fmt.format(across['mean_reported_when_chosen_on_the_scoring_set'])}"
                  f"   chosen elsewhere "
                  f"{fmt.format(across['mean_reported_when_chosen_on_a_separate_set'])}"
                  f"   never worse: {across['minimum_overstatement'] >= 0}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=int, default=N_POLICIES)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.policies, arguments.out))
