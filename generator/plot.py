"""Plot what training logs: losses, classifier accuracy and confidence per epoch, per-digit results, confusion.

Usage (from the repository root)::

    uv run python -m generator.plot --run ddpm                          # every panel the run has data for
    uv run python -m generator.plot --run ddpm --plot loss --plot confidence -o figures/ddpm.pdf
    uv run python -m generator.plot --run ddpm --run ddpm-small --plot accuracy   # compare runs
    uv run python -m generator.plot --run ddpm16+ddpm16-classifier      # join a run and its resumed continuation
    uv run python -m generator.plot --run ddpm --separate --format svg  # one file per panel, for slides
    uv run python -m generator.plot --run ddpm -p digit-bars --separate # a bar chart for every scored epoch
    uv run python -m generator.plot --run ddpm -p digit-bars --epoch 1 10 20 --columns 3
    uv run python -m generator.plot --run ddpm -p digit-bars-stacked --epoch 5 15 25 --bar-value confidence
    uv run python -m generator.plot --run ddpm --list                   # what can be plotted

Runs are read from outputs/metrics/<run>.jsonl (per-epoch losses and classifier summary) and
outputs/metrics/<run>_classifier.jsonl (one row per scored sample). Panels are the named plots below, or any
numeric per-epoch metric by its key (for example best_validation_loss). The file type follows --format or the
output's extension: png, pdf, svg.
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # write files only; never open a window

import matplotlib.pyplot as plt  # after choosing the backend
import numpy as np
from matplotlib.ticker import MaxNLocator

from generator.training import OUTPUT_DIR, ensure_outside_data

# Mean classifier confidence on 2,000 real validation digits: what a perfect generator would score
REAL_DIGIT_CONFIDENCE = 0.91
DIGITS = range(10)


@dataclass
class Run:
    name: str
    epochs: dict[int, dict] = field(default_factory=dict)  # epoch -> that epoch's metrics row
    samples: list[dict] = field(default_factory=list)  # one row per classified sample

    def series(self, key: str) -> tuple[list[int], list[float]]:
        """(epochs, values) for the epochs that logged `key`."""
        points = [(e, row[key]) for e, row in sorted(self.epochs.items()) if row.get(key) is not None]
        return [e for e, _ in points], [v for _, v in points]

    def sample_epochs(self) -> list[int]:
        return sorted({s["epoch"] for s in self.samples})

    def numeric_keys(self) -> list[str]:
        keys = {k for row in self.epochs.values() for k, v in row.items() if isinstance(v, int | float)}
        return sorted(keys - {"epoch", "step"})


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_run(spec: str, metrics_dir: Path) -> Run:
    """A run by name, or several joined with '+' (later parts win where epochs repeat, as after a resume)."""
    run = Run(spec)
    for name in spec.split("+"):
        epochs = read_jsonl(metrics_dir / f"{name}.jsonl")
        if not epochs:
            raise SystemExit(f"No metrics for run {name!r} in {metrics_dir}")
        run.epochs |= {row["epoch"]: row for row in epochs}
        samples = read_jsonl(metrics_dir / f"{name}_classifier.jsonl")
        replaced = {s["epoch"] for s in samples}
        run.samples = [s for s in run.samples if s["epoch"] not in replaced] + samples
    return run


def epoch_axis(ax, ylabel: str):
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)


def colours(runs: list[Run]) -> dict[str, str]:
    return {run.name: f"C{i}" for i, run in enumerate(runs)}


# Above this many epochs the markers merge into a thick smear, so lines are drawn plain
MARKER_LIMIT = 30


def line_style(runs: list[Run], key: str = "validation_loss") -> dict:
    """Markers on each epoch for short runs; thin plain lines once there are too many epochs to tell dots apart."""
    longest = max((len(run.series(key)[0]) for run in runs), default=0)
    return {"marker": "o", "markersize": 3} if longest <= MARKER_LIMIT else {"linewidth": 1.2}


def plot_loss(ax, runs: list[Run], **_):
    style = line_style(runs)
    for i, run in enumerate(runs):
        # Each line its own colour; the dash also marks validation, for greyscale prints
        for j, (key, dash, label) in enumerate((("train_loss", "-", "train"), ("validation_loss", "--", "validation"))):
            x, y = run.series(key)
            name = f"{run.name} {label}" if len(runs) > 1 else label
            ax.plot(x, y, dash, color=f"C{2 * i + j}", label=name, **style)
    ax.set_title("Denoising loss")
    epoch_axis(ax, "MSE")
    ax.legend()


def plot_accuracy(ax, runs: list[Run], **_):
    colour, style = colours(runs), line_style(runs, "classifier_accuracy")
    for run in runs:
        ax.plot(*run.series("classifier_accuracy"), color=colour[run.name], label=run.name, **style)
    ax.set_title("Classifier accuracy on generated digits")
    epoch_axis(ax, "fraction classified as the requested digit")
    ax.set_ylim(0, 1.02)
    ax.legend()


def plot_confidence(ax, runs: list[Run], **_):
    style = line_style(runs, "classifier_confidence")
    for i, run in enumerate(runs):
        prefix = f"{run.name} " if len(runs) > 1 else ""
        ax.plot(
            *run.series("classifier_confidence"), color=f"C{2 * i}", label=f"{prefix}confidence (top class)", **style
        )
        ax.plot(
            *run.series("classifier_target_probability"), "--", color=f"C{2 * i + 1}",
            label=f"{prefix}probability of requested digit", **style,
        )  # fmt: skip
    ax.axhline(REAL_DIGIT_CONFIDENCE, color="grey", linestyle=":", label=f"real MNIST digits ({REAL_DIGIT_CONFIDENCE})")
    ax.set_title("Classifier confidence on generated digits")
    epoch_axis(ax, "mean probability")
    ax.set_ylim(0, 1.02)
    ax.legend()


def per_digit(ax, run: Run, value, title: str, ylabel: str):
    """One line per digit of value(samples of that digit) over the epochs that were scored."""
    epochs, grid = digit_grid(run, value)
    cmap = plt.get_cmap("tab10")
    style = {"marker": "o", "markersize": 3} if len(epochs) <= MARKER_LIMIT else {"linewidth": 1}
    for digit in DIGITS:
        ax.plot(epochs, grid[digit], color=cmap(digit), label=str(digit), **style)
    ax.set_title(f"{title} ({run.name})")
    epoch_axis(ax, ylabel)
    ax.set_ylim(0, 1.02)
    ax.legend(title="digit", ncol=2, fontsize="small")


def digit_grid(run: Run, value) -> tuple[list[int], np.ndarray]:
    """Scored epochs, and value(samples) for each (digit, epoch) as a (10, epochs) array (NaN where none)."""
    epochs = run.sample_epochs()
    grid = np.full((len(DIGITS), len(epochs)), np.nan)
    for j, epoch in enumerate(epochs):
        rows = [s for s in run.samples if s["epoch"] == epoch]
        for digit in DIGITS:
            if chosen := [s for s in rows if s["target"] == digit]:
                grid[digit, j] = value(chosen)
    return epochs, grid


def digit_map(ax, run: Run, value, title: str, label: str):
    """Digits down, epochs across, coloured by value: readable however many epochs there are."""
    epochs, grid = digit_grid(run, value)
    image = ax.imshow(
        grid, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest",
        extent=(epochs[0] - 0.5, epochs[-1] + 0.5, len(DIGITS) - 0.5, -0.5),
    )  # fmt: skip
    ax.figure.colorbar(image, ax=ax, label=label)
    ax.set_yticks(DIGITS)
    ax.set_ylabel("requested digit")
    ax.set_xlabel("epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"{title} ({run.name})")


def mean_correct(samples):
    return np.mean([r["correct"] for r in samples])


def mean_confidence(samples):
    return np.mean([r["confidence"] for r in samples])


def plot_digit_accuracy(ax, run: Run, **_):
    per_digit(ax, run, mean_correct, "Accuracy per digit", "fraction correct")


def plot_digit_confidence(ax, run: Run, **_):
    per_digit(ax, run, mean_confidence, "Confidence per digit", "mean probability")


def plot_digit_accuracy_map(ax, run: Run, **_):
    digit_map(ax, run, mean_correct, "Accuracy per digit", "fraction correct")


def plot_digit_confidence_map(ax, run: Run, **_):
    digit_map(ax, run, mean_confidence, "Confidence per digit", "mean confidence")


def epoch_samples(run: Run, epoch: int) -> list[dict]:
    rows = [s for s in run.samples if s["epoch"] == epoch]
    if not rows:
        raise SystemExit(f"Run {run.name!r} has no classifier samples for epoch {epoch}; it has {run.sample_epochs()}")
    return rows


# --bar-value -> (per-sample value averaged per digit, axis label, title)
BAR_VALUES = {
    "accuracy": ("correct", "fraction classified correctly", "Accuracy per digit"),
    "confidence": ("confidence", "mean confidence (top class)", "Classifier confidence per digit"),
    "target-probability": ("target_probability", "mean probability of the requested digit", "Probability per digit"),
}


def digit_means(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """(mean of `key` per requested digit, sample count per digit); digits with no samples get 0."""
    sums, total = np.zeros(10), np.zeros(10, dtype=int)
    for s in rows:
        total[s["target"]] += 1
        sums[s["target"]] += s[key]
    return np.divide(sums, total, out=np.zeros(10), where=total > 0), total


def bar_axes(ax, value: str, title: str, run: Run):
    ax.set_xticks(DIGITS)
    ax.set_xlabel("requested digit")
    ax.set_ylabel(BAR_VALUES[value][1])
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{title}\n{run.name}", fontsize="medium")  # run names can be long, so on a line of their own


def plot_digit_bars(ax, run: Run, epoch: int, value: str = "accuracy", **_):
    """One epoch's value per requested digit, with the previous scored epoch outlined for comparison."""
    key = BAR_VALUES[value][0]
    rows = epoch_samples(run, epoch)
    heights, total = digit_means(rows, key)
    ax.bar(DIGITS, heights, color="C0", label=f"epoch {epoch}")
    top = heights.copy()
    earlier = [e for e in run.sample_epochs() if e < epoch]
    if earlier:
        prev, _ = digit_means(epoch_samples(run, earlier[-1]), key)
        ax.bar(DIGITS, prev, fill=False, edgecolor="black", linewidth=0.8, label=f"epoch {earlier[-1]}")
        top = np.maximum(top, prev)
    for digit in DIGITS:
        if total[digit]:
            text = (
                f"{round(heights[digit] * total[digit])}/{total[digit]}"
                if value == "accuracy"
                else f"{heights[digit]:.2f}"
            )
            ax.text(digit, top[digit] + 0.01, text, ha="center", va="bottom", fontsize="x-small")
    overall = np.mean([s[key] for s in rows])
    ax.axhline(overall, color="grey", linestyle="--", linewidth=1, label="all digits")
    bar_axes(ax, value, f"{BAR_VALUES[value][2]}, epoch {epoch}", run)
    ax.legend(loc="lower right", fontsize="small")


def epoch_colours(epochs: list[int]) -> dict[int, tuple]:
    """Colour follows time: a viridis ramp from dark purple (earliest chosen epoch) to yellow-green (latest).

    Epochs are spaced evenly along the ramp by their order, not their number, so 1, 2, 3, 5, 120 still get
    clearly different shades. The palest end of viridis is left out so the last epoch stays visible on white.
    """
    ramp = plt.get_cmap("viridis")(np.linspace(0, 0.85, len(epochs)) if len(epochs) > 1 else [0.45])
    return {e: tuple(c) for e, c in zip(sorted(epochs), ramp, strict=True)}


def plot_digit_bars_stacked(ax, run: Run, epochs: list[int], value: str = "accuracy", **_):
    """Several epochs overlapped on one bar chart, coloured dark (early) to light (late).

    For every digit the tallest bar is drawn at the back and the shortest at the front, each a little narrower, so
    every height stays visible. Epochs with the same height (common with 16 samples per digit) share one bar, split
    into side-by-side stripes in epoch order, so a tie never hides an epoch behind another.
    """
    key = BAR_VALUES[value][0]
    epochs = sorted(epochs)
    heights = {e: digit_means(epoch_samples(run, e), key)[0] for e in epochs}
    colour = epoch_colours(epochs)
    for digit in DIGITS:
        ties: dict[float, list[int]] = {}
        for epoch in epochs:  # in epoch order, so stripes run early (left) to late (right)
            ties.setdefault(round(float(heights[epoch][digit]), 6), []).append(epoch)
        levels = sorted(ties, reverse=True)
        widths = np.linspace(0.8, 0.5, len(levels)) if len(levels) > 1 else [0.8]
        for depth, height in enumerate(levels):
            tied, width = ties[height], widths[depth]
            stripe = width / len(tied)
            for i, epoch in enumerate(tied):
                ax.bar(digit - width / 2 + (i + 0.5) * stripe, height, width=stripe, color=colour[epoch],
                       edgecolor="white", linewidth=0.5, zorder=2 + depth)  # fmt: skip
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[e]) for e in epochs]
    ax.legend(handles, [f"epoch {e}" for e in epochs], loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small",
              ncol=1 + len(epochs) // 13)  # fmt: skip
    ax.grid(axis="y", alpha=0.3, zorder=0)
    shown = ", ".join(map(str, epochs)) if len(epochs) <= 6 else f"{epochs[0]} to {epochs[-1]}"
    bar_axes(ax, value, f"{BAR_VALUES[value][2]}, epochs {shown}", run)


def plot_confusion(ax, run: Run, epoch: int, **_):
    rows = epoch_samples(run, epoch)
    matrix = np.zeros((10, 10), dtype=int)
    for s in rows:
        matrix[s["target"], s["predicted"]] += 1
    ax.imshow(matrix, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        if count:
            colour = "white" if count > matrix.max() / 2 else "black"
            ax.text(j, i, count, ha="center", va="center", color=colour, fontsize="small")
    ax.set_xticks(DIGITS)
    ax.set_yticks(DIGITS)
    ax.set_xlabel("classifier's guess")
    ax.set_ylabel("requested digit")
    ax.set_title(f"Confusion, epoch {epoch} ({run.name})")


def plot_metric(ax, runs: list[Run], key: str, **_):
    colour = colours(runs)
    for run in runs:
        ax.plot(*run.series(key), color=colour[run.name], label=run.name, **line_style(runs, key))
    ax.set_title(key)
    epoch_axis(ax, key)
    ax.legend()


# name -> (draw, kind, help). kind "epochs": all runs on one panel, x is the epoch; "samples": one panel per run;
# "latest" and "each": one panel per run per --epoch, defaulting to the latest scored epoch or to every scored epoch;
# "together": one panel per run showing all of --epoch (default: every scored epoch)
PANELS = {
    "loss": (plot_loss, "epochs", "train and validation denoising loss"),
    "accuracy": (plot_accuracy, "epochs", "fraction of samples classified as the requested digit"),
    "confidence": (plot_confidence, "epochs", "mean classifier confidence and probability of the requested digit"),
    "digit-accuracy": (plot_digit_accuracy, "samples", "accuracy for each digit, one line per digit"),
    "digit-confidence": (plot_digit_confidence, "samples", "mean confidence for each digit"),
    "digit-accuracy-map": (plot_digit_accuracy_map, "samples", "accuracy as a digit-by-epoch heatmap (long runs)"),
    "digit-confidence-map": (plot_digit_confidence_map, "samples", "mean confidence as a digit-by-epoch heatmap"),
    "confusion": (plot_confusion, "latest", "requested digit against the classifier's guess, per --epoch"),
    "digit-bars": (plot_digit_bars, "each", "bar chart per digit (--bar-value), one per epoch (or per --epoch)"),
    "digit-bars-stacked": (
        plot_digit_bars_stacked,
        "together",
        "the --epoch epochs (default: all) overlapped on one bar chart",
    ),
}
DEFAULT_PANELS = ["loss", "accuracy", "confidence", "digit-accuracy", "confusion"]


def build_panels(
    names: list[str], runs: list[Run], epochs: list[int] | None, value: str = "accuracy"
) -> list[tuple[str, callable]]:
    """(file-name suffix, draw(ax)) for each panel to draw; per-sample panels repeat for each run that has samples."""
    available = {key for run in runs for key in run.numeric_keys()}
    panels = []
    for name in names:
        if name in PANELS:
            draw, needs, _ = PANELS[name]
            if needs == "epochs":
                panels.append((name, lambda ax, draw=draw: draw(ax, runs)))
                continue
            for run in (r for r in runs if r.samples):
                suffix = name if len(runs) == 1 else f"{name}-{run.name}"
                if needs == "samples":
                    panels.append((suffix, lambda ax, draw=draw, run=run: draw(ax, run)))
                    continue
                if needs == "together":
                    chosen = epochs or run.sample_epochs()
                    panels.append((suffix, lambda ax, draw=draw, run=run, e=chosen: draw(ax, run, e, value=value)))
                    continue
                default = run.sample_epochs()[-1:] if needs == "latest" else run.sample_epochs()
                panels.extend(
                    (
                        f"{suffix}-e{epoch:03d}",
                        lambda ax, draw=draw, run=run, epoch=epoch: draw(ax, run, epoch, value=value),
                    )
                    for epoch in epochs or default
                )
        elif name in available:
            panels.append((name, lambda ax, key=name: plot_metric(ax, runs, key)))
        else:
            raise SystemExit(f"Unknown plot {name!r}; see --list")
    return panels


def figure(panels, columns: int, panel_size: tuple[float, float], title: str | None):
    columns = min(columns, len(panels))
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(panel_size[0] * columns, panel_size[1] * rows), squeeze=False, layout="constrained"
    )
    for ax, (_, draw) in zip(axes.flat, panels, strict=False):
        draw(ax)
    # Line up every epoch axis on the same range, so panels with fewer scored epochs still align with the loss
    epoch_axes = [ax for ax in axes.flat[: len(panels)] if ax.get_xlabel() == "epoch"]
    if epoch_axes:
        low = min(ax.dataLim.x0 for ax in epoch_axes)
        high = max(ax.dataLim.x1 for ax in epoch_axes)
        for ax in epoch_axes:
            ax.set_xlim(low - 0.5, high + 0.5)
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    return fig


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="run name; repeat to compare, join with '+'")
    parser.add_argument("-p", "--plot", action="append", help="panel or metric key; repeat for several (default: all)")
    parser.add_argument("-o", "--output", type=Path, help="file to write (default: outputs/plots/<runs>.<format>)")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), help="file type (default: from --output, or png)")
    parser.add_argument("--separate", action="store_true", help="one file per panel: <output stem>-<panel>.<format>")
    parser.add_argument(
        "--epoch",
        type=int,
        nargs="+",
        help="epochs for confusion and the bar charts (default: the latest scored for confusion, all for the bars)",
    )
    parser.add_argument(
        "--bar-value",
        choices=BAR_VALUES,
        default="accuracy",
        help="what the bar charts show per digit (default: %(default)s)",
    )
    parser.add_argument("--columns", type=int, default=2, help="panels per row (default: %(default)s)")
    parser.add_argument("--size", type=float, nargs=2, default=(5.5, 4.0), metavar=("W", "H"), help="inches per panel")
    parser.add_argument("--dpi", type=int, default=200, help="resolution for png (default: %(default)s)")
    parser.add_argument("--title", help="title over the whole figure")
    parser.add_argument("--metrics-dir", type=Path, default=OUTPUT_DIR / "metrics")
    parser.add_argument("--list", action="store_true", help="list what can be plotted for these runs and exit")
    args = parser.parse_args(argv)

    runs = [load_run(spec, args.metrics_dir) for spec in args.run]
    if args.list:
        for name, (_, _, help_text) in PANELS.items():
            print(f"{name:20} {help_text}")
        print("\nper-epoch metric keys:", ", ".join(sorted({k for run in runs for k in run.numeric_keys()})))
        return

    has_samples = any(run.samples for run in runs)
    names = args.plot or [n for n in DEFAULT_PANELS if has_samples or n == "loss"]
    panels = build_panels(names, runs, args.epoch, args.bar_value)
    if not panels:
        raise SystemExit("Nothing to plot: these runs have no classifier samples")

    fmt = args.format or (args.output.suffix.lstrip(".") if args.output and args.output.suffix else "png")
    output = args.output or OUTPUT_DIR / "plots" / "_vs_".join(run.name for run in runs)
    output = output.with_suffix(f".{fmt}")
    ensure_outside_data(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    figures = [(f"-{suffix}", [(suffix, draw)]) for suffix, draw in panels] if args.separate else [("", panels)]
    for suffix, group in figures:
        fig = figure(group, args.columns, tuple(args.size), args.title)
        if re.fullmatch(rf"-{re.escape(output.stem)}(-e\d+)?", suffix):  # -o confusion.png -> confusion-e010.png
            suffix = suffix[len(output.stem) + 1 :]
        path = output.with_name(f"{output.stem}{suffix}{output.suffix}")
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
