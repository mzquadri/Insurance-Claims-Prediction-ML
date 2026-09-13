"""Fail if the README stops agreeing with the recorded results.

    python scripts/check_repository.py

Checks that the files the README points at exist and compile, that every number it
quotes still matches `results/benchmark.json`, and that the three committed
figures were drawn from that same file. A results file is easy to regenerate and
a README is easy to forget, and a repository whose headline numbers no longer
describe its own output is worse than one with no numbers at all.

Each claim is matched with its surrounding words included, so a value that has
drifted into a different sentence does not accidentally satisfy a check.

Run this before regenerating the figures, not after. It reads what is committed.
"""

from __future__ import annotations

import json
import py_compile
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark.json"
FIGURES = ROOT / "docs" / "figures"

#: tEXt key that scripts/figures/portfolio_style.py writes the source values to.
BENCHMARK_KEY = "Benchmark"

PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
NUL = bytes([0x00])

REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "src/synthetic_portfolio.py",
    "src/benchmark.py",
    "src/data_pipeline.py",
    "src/model_training.py",
    "src/calibration.py",
    "src/threshold_optimizer.py",
    "src/explainability.py",
    "tests/test_leakage.py",
    "tests/test_selection_path.py",
    "scripts/figures/generate_figures.py",
    "scripts/check_reproducibility.py",
    "notebooks/01_EDA_and_Feature_Engineering.ipynb",
    "notebooks/02_Model_Training_and_Evaluation.ipynb",
    "docs/figures/01_available_signal.png",
    "docs/figures/02_selection_optimism.png",
    "docs/figures/03_calibration.png",
)

MODELS = ("logistic_regression", "random_forest", "gradient_boosting")

#: Small counts read better spelled out in prose, but they still have to be tied
#: to the recorded value, so the word is derived from the number rather than
#: written twice. An unlisted value raises instead of silently passing.
WORDS = {9: "nine", 22: "twenty-two", 99: "ninety-nine"}


def spelled(value: int) -> str:
    if value not in WORDS:
        raise SystemExit(f"  {value} has no spelled form in check_repository.WORDS; "
                         f"add it, or write the digits in the README")
    return WORDS[value]
NAMES = {"logistic_regression": "Logistic regression",
         "random_forest": "Random forest",
         "gradient_boosting": "Gradient boosting"}


def claims(data: dict) -> list[tuple[str, str]]:
    """Every quoted number, paired with enough context to anchor it."""
    reference = data["reference_points"]
    models = data["models"]
    thresholds = data["thresholds"]
    calibration = data["calibration"]
    floor = reference["bayes_floor"]["brier"]
    uninformed = reference["brier_predicting_the_base_rate"]
    available = uninformed - floor
    rules = reference["business_value_of_fixed_rules"]

    out = [
        (f"generates {data['data']['policies']:,} policies", "the portfolio size"),
        (f"with {spelled(data['data']['features'])} features", "the feature count"),
        (f"rate of {data['data']['claim_rate'] * 100:.2f} percent", "the claim rate"),
        (f"searched {spelled(99)} thresholds", "the size of the threshold grid"),
        (f"Brier of {uninformed:.5f}", "the uninformed Brier"),
        (f"knew every true probability scores {floor:.5f}", "the Bayes floor"),
        (f"gap of {available:.5f}", "the available signal"),
        (f"{reference['accuracy_by_never_predicting_a_claim'] * 100:.2f} percent\naccurate",
         "the accuracy of never predicting a claim"),
    ]

    for name in MODELS:
        entry = models[name]["test"]
        captured = (uninformed - entry["brier"]) / available * 100
        out.append((
            f"| {NAMES[name]} | {entry['brier']:.5f} | {captured:.1f}% | "
            f"{entry['roc_auc']:.4f} |",
            f"the {NAMES[name].lower()} results row"))

        across = thresholds[name]["f1"]["across_repartitions"]
        out.append((
            f"| {NAMES[name]} | "
            f"{across['mean_reported_when_chosen_on_the_scoring_set']:.4f} | "
            f"{across['mean_reported_when_chosen_on_a_separate_set']:.4f} | "
            f"{across['mean_overstatement']:+.4f} |",
            f"the {NAMES[name].lower()} threshold row"))

    business = [thresholds[name]["business"]["across_repartitions"]["mean_overstatement"]
                for name in MODELS]
    out += [
        (f"from\n+{min(business):,.0f} to +{max(business):,.0f} units",
         "the business value range"),
        (f"gap reached {max(thresholds[name]['f1']['across_repartitions'][
            'worst_single_split_overstatement'] for name in MODELS):.3f}",
         "the worst single-split F1 gap"),
        (f"between {min(thresholds[name]['f1']['across_repartitions'][
            'fraction_of_splits_where_both_choices_tie'] for name in MODELS) * 100:.0f} "
         f"and {max(thresholds[name]['f1']['across_repartitions'][
             'fraction_of_splits_where_both_choices_tie'] for name in MODELS) * 100:.0f} "
         f"percent of splits", "the tie rate"),

        (f"| Never flag a policy | {rules['never_flag_a_policy']:,.0f} |",
         "the never-flag row"),
        (f"| Flag every policy | {rules['flag_every_policy']:,.0f} |",
         "the flag-everything row"),
        (f"true probabilities | {rules['oracle_rule_on_the_true_probability']:,.0f} |",
         "the oracle row"),
        (f"threshold from validation | "
         f"{thresholds['logistic_regression']['business']['across_repartitions'][
             'mean_reported_when_chosen_on_a_separate_set']:,.0f} |",
         "the logistic regression business row"),

        (f"{calibration['random_forest']['refit_platt_auc_shift']:.5f} AUC for the "
         f"random forest", "the random forest AUC shift"),
        (f"{calibration['gradient_boosting']['refit_platt_auc_shift']:.5f} for\n"
         f"gradient boosting", "the gradient boosting AUC shift"),
        (f"changed AUC by exactly "
         f"{calibration['logistic_regression']['posthoc_platt_auc_shift']:.1f}",
         "the frozen-map AUC shift"),
    ]
    return out


def ratio_claims(data: dict) -> list[tuple[str, float, float]]:
    """Ratios the README states in words, recomputed rather than trusted."""
    reference = data["reference_points"]
    rules = reference["business_value_of_fixed_rules"]
    honest = data["thresholds"]["logistic_regression"]["business"][
        "across_repartitions"]["mean_reported_when_chosen_on_a_separate_set"]
    span = rules["flag_every_policy"] - rules["oracle_rule_on_the_true_probability"]
    return [
        ("about 42 percent of the distance",
         (rules["flag_every_policy"] - honest) / span * 100, 42.0),
        ("about three percent of the score",
         (reference["brier_predicting_the_base_rate"]
          - reference["bayes_floor"]["brier"])
         / reference["brier_predicting_the_base_rate"] * 100, 3.0),
    ]


def png_text(path: Path) -> dict[str, str]:
    """The tEXt entries of a PNG, read without a third-party imaging library."""
    raw = path.read_bytes()
    if raw[:len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise SystemExit(f"  {path.name} is not a PNG")
    entries, offset = {}, 8
    while offset + 8 <= len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        if kind == b"tEXt":
            key, _, value = raw[offset + 8:offset + 8 + length].partition(NUL)
            entries[key.decode("latin-1")] = value.decode("latin-1")
        elif kind == b"IEND":
            break
        offset += 12 + length
    return entries


def figure_claims(data: dict) -> list[str]:
    """Check the committed figures against the results they were drawn from.

    The figures carry numbers a reader can see, and until now nothing tied them
    to results/benchmark.json. Regenerating the benchmark and forgetting the
    figures left three images stating superseded values, and continuous
    integration was happy because it rendered them into the working tree and
    never looked at what it had replaced.

    Comparing bytes cannot do this. The figures use whichever of the fonts in
    portfolio_style.FONTS the machine provides, so the same data rendered here
    and on the Linux runner agree on every number and on none of the pixels.
    What is compared is the values each figure recorded when it was written.
    """
    failures = []
    present = sorted(path.name for path in FIGURES.glob("*.png"))
    expected = sorted(Path(name).name for name in REQUIRED_FILES
                      if name.startswith("docs/figures/"))
    if present != expected:
        failures.append(f"  docs/figures holds {present}, expected {expected}")

    renderers, checked = set(), 0
    for name in expected:
        path = FIGURES / name
        if not path.is_file():
            continue
        text = png_text(path)
        renderers.add(text.get("Software", "unrecorded"))
        if BENCHMARK_KEY not in text:
            failures.append(
                f"  {name} records no source values; rerun "
                f"scripts/figures/generate_figures.py")
            continue
        for path_in_results, drawn in json.loads(text[BENCHMARK_KEY]).items():
            node = data
            for key in path_in_results.split("/"):
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if node is None:
                failures.append(f"  {name} was drawn from {path_in_results}, "
                                f"which the results no longer contain")
            elif node != drawn:
                failures.append(f"  {name} shows {path_in_results} as {drawn}, "
                                f"the results now say {node}")
            checked += 1

    # A figure regenerated on its own carries a different matplotlib version from
    # the rest as soon as the pinned version moves, which is what a half-finished
    # regeneration looks like.
    if len(renderers) > 1:
        failures.append(f"  the figures were not rendered together: {sorted(renderers)}")

    print(f"  {checked} values behind {len(expected)} figures match the results")
    return failures


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"  missing required files: {', '.join(missing)}")

    for source in sorted((ROOT / "src").glob("*.py")):
        py_compile.compile(source, doraise=True)
    print(f"  {len(REQUIRED_FILES)} required files present, src/ compiles")

    if not RESULTS.exists():
        raise SystemExit("  results/benchmark.json is missing; run python -m src.benchmark")

    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))

    failures = []
    checks = claims(data)
    for expected, description in checks:
        if re.sub(r"\s+", " ", expected) not in flat:
            failures.append(f"  README does not state {description}: expected "
                            f"{re.sub(r'\\s+', ' ', expected).strip()!r}")
    print(f"  {len(checks) - len(failures)} of {len(checks)} recorded numbers "
          f"found in the README")

    failures += figure_claims(data)

    ratios = ratio_claims(data)
    for phrase, actual, stated in ratios:
        if abs(actual - stated) / stated > 0.08:
            failures.append(f"  README says {phrase!r} but the recorded ratio is "
                            f"{actual:.2f}, not {stated}")
        if re.sub(r"\s+", " ", phrase) not in flat:
            failures.append(f"  README no longer contains the phrase {phrase!r}")
    print(f"  {len(ratios)} stated ratios recomputed from the results")

    if failures:
        print()
        for failure in failures:
            print(failure)
        raise SystemExit(f"\n  {len(failures)} claims in the README or the "
                         f"figures no longer match results/benchmark.json")

    print("  README and results/benchmark.json agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
