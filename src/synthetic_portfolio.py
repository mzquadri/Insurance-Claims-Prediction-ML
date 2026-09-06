"""Generate a policy portfolio whose true claim probability is known.

    python -m src.synthetic_portfolio

This is not insurance data and carries no actuarial meaning. Every relationship
below was written here, and the effect sizes were chosen to make a demonstration
legible rather than to describe any real portfolio.

It exists because of one thing real data cannot give. Calibration is a claim about
predicted probabilities matching true ones, and on real data the true probability
of a single policy is never observable, so calibration can only be checked against
frequencies in bins. Here the probability that generated each outcome is recorded
alongside it, so a predicted probability can be compared with the quantity it is
supposed to estimate, and the best score any model could reach is computable
instead of assumed.

The column `true_probability` is that quantity. It is never a feature. Handing it
to a model would be the most direct leak available in this repository, and
`tests/test_leakage.py` asserts it stays out of the feature matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
N_POLICIES = 40_000

#: Intercept chosen so the portfolio claims at roughly six percent a year, which is
#: the order of magnitude of the public dataset this repository was written around.
#: The imbalance is the point: it is what makes accuracy useless and thresholds
#: worth choosing.
INTERCEPT = -3.24

#: Columns that describe the generating process rather than the policy. None of
#: them may reach a model.
NOT_FEATURES = ("is_claim", "true_probability")


def true_log_odds(frame: pd.DataFrame) -> np.ndarray:
    """The relationship the outcomes are drawn from, written out in full.

    Two properties are deliberate. It is nonlinear in driver age, so a logistic
    regression cannot represent it exactly and there is something for a tree to
    win. And it contains one interaction, between young drivers and powerful cars,
    so an additive model leaves a little on the table.
    """
    age = frame["driver_age"].to_numpy(dtype=float)
    young = np.clip(25.0 - age, 0, None) / 7.0          # steep below 25
    elderly = np.clip(age - 70.0, 0, None) / 10.0       # rises again late
    mileage = np.log(frame["annual_mileage_km"].to_numpy(dtype=float) / 12_000.0)
    power = (frame["engine_power_kw"].to_numpy(dtype=float) - 110.0) / 45.0

    return (
        INTERCEPT
        + 0.85 * young
        + 0.40 * elderly
        + 0.55 * mileage
        + 0.62 * frame["prior_claims"].to_numpy(dtype=float)
        + 0.33 * frame["region_density"].to_numpy(dtype=float)
        + 0.20 * power
        + 0.05 * frame["vehicle_age_years"].to_numpy(dtype=float)
        - 0.09 * frame["policy_tenure_years"].to_numpy(dtype=float)
        + 0.30 * young * np.clip(power, 0, None)        # the one interaction
    )


def generate_portfolio(n_policies: int = N_POLICIES, seed: int = SEED) -> pd.DataFrame:
    """Draw policies, compute each one's true claim probability, then draw outcomes."""
    rng = np.random.default_rng(seed)

    driver_age = np.clip(rng.gamma(shape=9.0, scale=4.4, size=n_policies) + 18, 18, 85)
    frame = pd.DataFrame({
        "driver_age": driver_age.round(0),
        "vehicle_age_years": np.clip(rng.exponential(4.2, n_policies), 0, 20).round(1),
        "engine_power_kw": np.clip(rng.normal(110, 32, n_policies), 45, 260).round(0),
        "annual_mileage_km": np.clip(
            rng.lognormal(np.log(12_000), 0.42, n_policies), 1_500, 70_000).round(-2),
        "region_density": rng.choice([0.0, 1.0, 2.0], n_policies, p=[0.34, 0.41, 0.25]),
        "prior_claims": rng.poisson(0.28, n_policies).clip(0, 4),
        "policy_tenure_years": np.clip(rng.exponential(3.6, n_policies), 0, 25).round(1),
    })

    # Two columns with no bearing on the outcome at all. Feature attribution that
    # ranks either of them highly is telling you about the model, not the risk.
    frame["marketing_channel"] = rng.choice([0.0, 1.0, 2.0], n_policies)
    frame["policy_document_version"] = rng.integers(1, 5, n_policies).astype(float)

    probability = 1.0 / (1.0 + np.exp(-true_log_odds(frame)))
    frame["true_probability"] = probability
    frame["is_claim"] = (rng.random(n_policies) < probability).astype(int)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Everything a model is allowed to see."""
    return [column for column in frame.columns if column not in NOT_FEATURES]


def bayes_limits(probability: np.ndarray, outcome: np.ndarray) -> dict:
    """What a model that knew every true probability would actually score here.

    A policy with a true probability of 0.06 produces a claim six percent of the
    time no matter how good the model is, so no model can drive these to zero.
    This is the floor a reported Brier score should be read against.

    Scored on the outcomes that were actually drawn, not on their expectation.
    The expected value, mean p(1-p), differs from the realised score by sampling
    noise, and comparing an expectation with a realisation made the floor look
    worse than predicting the base rate, which is impossible.
    """
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    safe = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "brier": float(((outcome - probability) ** 2).mean()),
        "log_loss": float(-(outcome * np.log(safe)
                            + (1 - outcome) * np.log(1 - safe)).mean()),
        "expected_brier": float((probability * (1 - probability)).mean()),
        "mean_true_probability": float(probability.mean()),
        "observed_claim_rate": float(outcome.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=int, default=N_POLICIES)
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()

    frame = generate_portfolio(arguments.policies, arguments.seed)
    limits = bayes_limits(frame["true_probability"], frame["is_claim"])

    print(f"  {len(frame):,} policies, {len(feature_columns(frame))} features")
    print(f"  claims: {frame['is_claim'].sum():,} "
          f"({frame['is_claim'].mean() * 100:.2f} percent of the portfolio)")
    print(f"  true probability spans {frame['true_probability'].min():.4f} "
          f"to {frame['true_probability'].max():.4f}")
    print("\n  the best any model could do, knowing every true probability:")
    print(f"    Brier    {limits['brier']:.5f}")
    print(f"    log loss {limits['log_loss']:.5f}")
    print("\n  for comparison, predicting the base rate for every policy:")
    rate = frame["is_claim"].mean()
    print(f"    Brier    {float(((frame['is_claim'] - rate) ** 2).mean()):.5f}")
    print(f"    accuracy {max(1 - rate, rate) * 100:.2f} percent, by never "
          f"predicting a claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
