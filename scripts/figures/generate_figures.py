"""Render the three figures in the README from results/benchmark.json.

    python scripts/figures/generate_figures.py

Each figure answers one question:

  01  how much signal is there to win, and how much does each model win
  02  what does choosing a threshold on the set you report it on buy you
  03  what does calibration actually change
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import portfolio_style as ps

RESULTS = ROOT / "results" / "benchmark.json"
FIGURES = ROOT / "docs" / "figures"

LABELS = {
    "logistic_regression": "Logistic\nregression",
    "random_forest": "Random\nforest",
    "gradient_boosting": "Gradient\nboosting",
}


def load() -> dict:
    if not RESULTS.exists():
        raise SystemExit("run `python -m src.benchmark` first")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


class Tap:
    """Reads values out of the results and remembers which ones it handed over.

    Every number in these figures comes through here, so what gets stamped into
    the PNG is what the figure was actually drawn from rather than a second list
    that has to be kept in step by hand. A value the figure stops reading stops
    being stamped, and a value it starts reading is stamped without anyone
    remembering to add it.
    """

    def __init__(self, data: dict) -> None:
        self._data = data
        self.read: dict[str, float] = {}

    def __call__(self, *path: str) -> float:
        node = self._data
        for key in path:
            node = node[key]
        self.read["/".join(path)] = node
        return node


def figure_01_available_signal(data: dict) -> None:
    """How much is there to win, and how much does each model win?"""
    take = Tap(data)
    floor = take("reference_points", "bayes_floor", "brier")
    uninformed = take("reference_points", "brier_predicting_the_base_rate")
    available = uninformed - floor

    names = list(LABELS)
    captured = [(uninformed - take("models", name, "test", "brier")) / available
                for name in names]

    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    fig.subplots_adjust(left=0.155, right=0.955, top=0.745, bottom=0.235)

    bars = ax.bar(np.arange(len(names)), [value * 100 for value in captured],
                  color=[ps.BLUE, ps.SLATE, ps.SLATE], width=0.54)
    for bar, value in zip(bars, captured, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 100 + 3.0,
                f"{value * 100:.1f}%", ha="center", fontsize=12,
                color=ps.INK, fontweight="600")

    ax.axhline(100, color=ps.GREEN, linewidth=1.6, linestyle=(0, (5, 3)))
    ax.text(len(names) - 0.46, 103.0,
            "a model that knew every true probability", fontsize=10,
            color=ps.GREEN, ha="right")
    ax.axhline(0, color=ps.INK, linewidth=1.1)
    ax.text(len(names) - 0.46, -7.0, "predicting the base rate for every policy",
            fontsize=10, color=ps.MUTED, ha="right", va="top")

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([LABELS[name] for name in names], fontsize=11, color=ps.INK)
    ax.set_ylabel("Share of the available signal captured", fontsize=11, color=ps.MUTED)
    ax.set_ylim(-13, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ps.clean(ax, grid_axis="y")

    ps.title_block(
        fig, "The simplest model captures almost all of it",
        "Claim probability on a generated portfolio. The scale runs from predicting "
        "one number for everybody\nto knowing the probability behind every policy.")

    ps.footnote(fig, [
        f"That whole scale is narrow: in Brier score it runs from {uninformed:.5f} "
        f"down to {floor:.5f}, a range of {available:.5f}, so the three",
        "models sit within half a percent of each other on the raw metric. The "
        "generating process is close to linear, which is why the",
        "logistic regression is hard to beat.",
    ], y=0.105)
    ps.save(fig, FIGURES, "01_available_signal", sources=take.read)


def figure_02_selection_optimism(data: dict) -> None:
    """What does choosing a threshold on the set you report it on buy you?"""
    take = Tap(data)
    entries = [
        (name, {
            "mean_reported_when_chosen_on_the_scoring_set": take(
                "thresholds", name, "f1", "across_repartitions",
                "mean_reported_when_chosen_on_the_scoring_set"),
            "mean_reported_when_chosen_on_a_separate_set": take(
                "thresholds", name, "f1", "across_repartitions",
                "mean_reported_when_chosen_on_a_separate_set"),
            "worst_single_split_overstatement": take(
                "thresholds", name, "f1", "across_repartitions",
                "worst_single_split_overstatement"),
            "repeats": take("thresholds", name, "f1", "across_repartitions",
                            "repeats"),
        })
        for name in LABELS]
    # Both counts in this figure are read rather than typed. The caption used to
    # say 600 and the axis 200, neither of which would have moved if the number
    # of repeats did.
    per_model = {entry["repeats"] for _, entry in entries}
    repeats = per_model.pop() if len(per_model) == 1 else None
    total = sum(entry["repeats"] for _, entry in entries)

    fig, ax = plt.subplots(figsize=(11.4, 6.4))
    fig.subplots_adjust(left=0.105, right=0.955, top=0.755, bottom=0.235)

    positions = np.arange(len(entries))
    width = 0.33
    chosen_here = [entry["mean_reported_when_chosen_on_the_scoring_set"]
                   for _, entry in entries]
    chosen_apart = [entry["mean_reported_when_chosen_on_a_separate_set"]
                    for _, entry in entries]

    ax.bar(positions - width / 2, chosen_here, width, color=ps.AMBER,
           label="threshold chosen on the set it is reported on")
    ax.bar(positions + width / 2, chosen_apart, width, color=ps.BLUE,
           label="threshold chosen on a separate set")

    for index, (here, apart) in enumerate(zip(chosen_here, chosen_apart, strict=True)):
        ax.text(index - width / 2, here + 0.0025, f"{here:.4f}", ha="center",
                fontsize=10.4, color=ps.AMBER, fontweight="600")
        ax.text(index + width / 2, apart + 0.0025, f"{apart:.4f}", ha="center",
                fontsize=10.4, color=ps.BLUE, fontweight="600")
        # The gap is small in absolute F1 and the axis starts at zero, so it is
        # stated rather than made to look bigger by cropping the scale.
        ax.annotate("", xy=(index - width / 2, here), xytext=(index + width / 2, here),
                    arrowprops={"arrowstyle": "-", "color": ps.FAINT,
                                "linewidth": 0.9, "linestyle": (0, (2, 2))})
        ax.text(index, here + 0.0135, f"overstated by {here - apart:+.4f}",
                ha="center", fontsize=9.8, color=ps.MUTED)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[name] for name, _ in entries], fontsize=11,
                       color=ps.INK)
    ax.set_ylabel(
        "F1 on the reported set, mean over "
        + (f"{repeats} repartitions" if repeats else "the repartitions"),
        fontsize=11, color=ps.MUTED)
    ax.set_ylim(0, max(chosen_here) * 1.36)
    ax.legend(loc="upper right", frameon=False, fontsize=10.4, labelcolor=ps.MUTED)
    ps.clean(ax, grid_axis="y")

    ps.title_block(
        fig, "A threshold cannot be chosen and judged on the same data",
        "The model is held fixed. Only which half of the held-out rows picks the "
        "threshold changes.")

    worst = max(entry["worst_single_split_overstatement"] for _, entry in entries)
    ps.footnote(fig, [
        "The amber bar is not an estimate that came out high. The search returns "
        "whichever threshold maximises the quantity then",
        f"reported, so it can only tie or win: across {total} repartitions the gap "
        f"was never negative, and reached {worst:.3f} F1 at worst.",
    ])
    ps.save(fig, FIGURES, "02_selection_optimism", sources=take.read)


def figure_03_calibration(data: dict) -> None:
    """What does calibration actually change?"""
    take = Tap(data)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 6.6))
    fig.subplots_adjust(left=0.082, right=0.965, top=0.70, bottom=0.255, wspace=0.27)

    # Left: distance to the true probability, which real data cannot measure.
    methods = ("uncalibrated", "posthoc_isotonic", "isotonic")
    method_labels = ("no calibration",
                     "isotonic map on the frozen model",
                     "CalibratedClassifierCV(cv=5)")
    colours = (ps.SLATE, ps.BLUE, ps.GREEN)
    positions = np.arange(len(LABELS))
    width = 0.26

    for offset, (method, label, colour) in enumerate(
            zip(methods, method_labels, colours, strict=True)):
        values = [take("calibration", name, "methods", method, "test",
                       "mean_absolute_error_against_truth") * 100
                  for name in LABELS]
        left.bar(positions + (offset - 1) * width, values, width, color=colour,
                 label=label)

    left.set_xticks(positions)
    left.set_xticklabels([LABELS[name].replace("\n", " ") for name in LABELS],
                         fontsize=9.6, color=ps.INK)
    left.set_ylabel("Mean |predicted - true| probability, points", fontsize=10.6,
                    color=ps.MUTED)
    left.set_title("Distance to the probability that generated the outcome",
                   fontsize=12, color=ps.INK, pad=11, loc="left")
    left.legend(frameon=False, fontsize=9.0, labelcolor=ps.MUTED, loc="upper left")
    left.set_ylim(0, 2.35)
    ps.clean(left, grid_axis="y")

    # Right: what each form of calibration does to the ranking.
    posthoc = [take("calibration", name, "posthoc_platt_auc_shift")
               for name in LABELS]
    refit = [take("calibration", name, "refit_platt_auc_shift")
             for name in LABELS]

    right.bar(positions - 0.17, posthoc, 0.32, color=ps.BLUE,
              label="Platt applied to the frozen model")
    right.bar(positions + 0.17, refit, 0.32, color=ps.AMBER,
              label="CalibratedClassifierCV(cv=5)")
    for index, value in enumerate(posthoc):
        right.text(index - 0.17, max(refit) * 0.022, "exactly 0" if value == 0
                   else f"{value:.0e}", ha="center", fontsize=9.4, color=ps.BLUE,
                   rotation=90, va="bottom")
    for index, value in enumerate(refit):
        right.text(index + 0.17, value + max(refit) * 0.03, f"{value:.1e}",
                   ha="center", fontsize=9.6, color=ps.AMBER, fontweight="600")

    right.set_xticks(positions)
    right.set_xticklabels([LABELS[name].replace("\n", " ") for name in LABELS],
                          fontsize=9.6, color=ps.INK)
    right.set_ylabel("Change in test AUC", fontsize=10.6, color=ps.MUTED)
    right.set_title("What each one does to the ranking", fontsize=12, color=ps.INK,
                    pad=11, loc="left")
    right.set_ylim(0, max(refit) * 1.35)
    right.legend(frameon=False, fontsize=9.8, labelcolor=ps.MUTED, loc="upper left")
    ps.clean(right, grid_axis="y")

    ps.title_block(
        fig, "What helps here is the averaging, not the calibration map",
        "Two things go by the same name. Only one of them is a map applied to a "
        "fitted model, and on this\ndata it makes every model worse.")

    ps.footnote(fig, [
        "An isotonic map fitted on validation and applied to the frozen model moves "
        "all three further from the truth. The scikit-learn",
        "call with an integer cv trains five fresh models and averages them, which "
        "helps the two tree models, and reorders them as the",
        "right panel shows. A map cannot reorder anything, so a gain that comes with "
        "a ranking change did not come from calibrating.",
    ], y=0.105)
    ps.save(fig, FIGURES, "03_calibration", sources=take.read)


def main() -> int:
    ps.apply()
    data = load()
    print(f"  reading {RESULTS.relative_to(ROOT).as_posix()}")
    figure_01_available_signal(data)
    figure_02_selection_optimism(data)
    figure_03_calibration(data)
    print(f"  figures in {FIGURES.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
