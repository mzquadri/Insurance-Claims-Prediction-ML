"""Rerun the benchmark and check that it still supports the same conclusions.

    python scripts/check_reproducibility.py

The recorded numbers were produced on one machine with one set of pinned library
versions. Requiring a rerun to match them digit for digit would make this fail on
another platform for reasons unrelated to the code being wrong: a different BLAS,
a different thread count, a different summation order.

So two standards are applied. The findings the README argues from are qualitative
and must hold exactly, because if one flips the README is saying something false.
The numbers are compared within a tolerance wide enough to absorb platform
arithmetic and narrow enough that a real regression fails it.

Writes nothing. The recorded results file is not overwritten.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "results" / "benchmark.json"

TOLERANCE = 0.05
MODELS = ("logistic_regression", "random_forest", "gradient_boosting")


def findings(data: dict) -> dict:
    """The qualitative claims the README rests on. Each must survive a rerun."""
    reference = data["reference_points"]
    models = data["models"]
    thresholds = data["thresholds"]
    calibration = data["calibration"]

    floor = reference["bayes_floor"]["brier"]
    uninformed = reference["brier_predicting_the_base_rate"]
    rules = reference["business_value_of_fixed_rules"]

    def captured(name: str) -> float:
        return (uninformed - models[name]["test"]["brier"]) / (uninformed - floor)

    out = {
        "the floor is better than predicting the base rate": floor < uninformed,
        "no model beats the floor": all(
            models[name]["test"]["brier"] > floor for name in MODELS),
        "logistic regression captures the most available signal": (
            captured("logistic_regression") > captured("random_forest")
            and captured("logistic_regression") > captured("gradient_boosting")),
        "logistic regression captures more than nine tenths of it":
            captured("logistic_regression") > 0.9,
        "never predicting a claim is over ninety percent accurate":
            reference["accuracy_by_never_predicting_a_claim"] > 0.9,
        "flagging every policy beats flagging none":
            rules["flag_every_policy"] > rules["never_flag_a_policy"],
        "the oracle rule beats flagging every policy":
            rules["oracle_rule_on_the_true_probability"] > rules["flag_every_policy"],
    }

    for name in MODELS:
        for objective in ("f1", "business"):
            across = thresholds[name][objective]["across_repartitions"]
            out[f"{name}, {objective}: choosing on the scoring set never scores "
                f"worse there"] = across["minimum_overstatement"] >= 0.0
            out[f"{name}, {objective}: and on average it scores better"] = (
                across["mean_overstatement"] > 0.0)

        methods = calibration[name]["methods"]
        out[f"{name}: a frozen monotone map leaves the ranking exactly alone"] = (
            calibration[name]["posthoc_platt_auc_shift"] == 0.0)
        out[f"{name}: the isotonic map on the frozen model does not beat leaving "
            f"it alone"] = (
                methods["posthoc_isotonic"]["test"]["mean_absolute_error_against_truth"]
                >= methods["uncalibrated"]["test"]["mean_absolute_error_against_truth"])

    out["refitting and averaging reorders the tree models"] = all(
        calibration[name]["refit_platt_auc_shift"] > 0.0
        for name in ("random_forest", "gradient_boosting"))
    out["and it helps them, unlike the frozen map"] = all(
        calibration[name]["methods"]["isotonic"]["test"][
            "mean_absolute_error_against_truth"]
        < calibration[name]["methods"]["uncalibrated"]["test"][
            "mean_absolute_error_against_truth"]
        for name in ("random_forest", "gradient_boosting"))
    return out


def values(data: dict) -> dict:
    """The numbers compared with a tolerance."""
    out = {f"brier.{name}": data["models"][name]["test"]["brier"] for name in MODELS}
    out.update({f"auc.{name}": data["models"][name]["test"]["roc_auc"]
                for name in MODELS})
    out.update({"floor": data["reference_points"]["bayes_floor"]["brier"],
                "uninformed": data["reference_points"]["brier_predicting_the_base_rate"]})
    for name in MODELS:
        across = data["thresholds"][name]["f1"]["across_repartitions"]
        out[f"threshold_gap.{name}"] = across["mean_overstatement"]
    return out


def main() -> int:
    if not RECORDED.exists():
        raise SystemExit("  results/benchmark.json is missing; run python -m src.benchmark")
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as directory:
        fresh_path = Path(directory) / "benchmark.json"
        print("  rerunning the benchmark into a temporary file")
        completed = subprocess.run(
            [sys.executable, "-m", "src.benchmark", "--out", str(fresh_path)],
            cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            print(completed.stdout[-2000:])
            print(completed.stderr[-2000:])
            raise SystemExit(f"  the benchmark failed with exit code {completed.returncode}")
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))

    failures = []
    recorded_findings, fresh_findings = findings(recorded), findings(fresh)
    for description, holds in recorded_findings.items():
        if not holds:
            failures.append(f"  the recorded results no longer support: {description}")
        elif not fresh_findings[description]:
            failures.append(f"  a rerun no longer supports: {description}")
    held = sum(1 for key in recorded_findings
               if recorded_findings[key] and fresh_findings[key])
    print(f"  {len(recorded_findings)} findings checked, {held} hold")

    recorded_values, fresh_values = values(recorded), values(fresh)
    drifted = 0
    for key, was in recorded_values.items():
        now = fresh_values[key]
        if was and abs(now - was) / abs(was) > TOLERANCE:
            drifted += 1
            failures.append(f"  {key} moved from {was:.5f} to {now:.5f}, "
                            f"more than {TOLERANCE:.0%}")
    print(f"  {len(recorded_values)} values compared, {len(recorded_values) - drifted} "
          f"within {TOLERANCE:.0%}")

    if failures:
        print()
        for failure in failures:
            print(failure)
        raise SystemExit(f"\n  {len(failures)} checks failed")

    print("  the rerun supports the same conclusions as the recorded results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
