"""Score saved checkpoints with the pretrained digit classifier, in the same format training logs each epoch.

Usage (from the repository root)::

    uv run python -m generator.score --checkpoint outputs/checkpoints/ddpm/epoch-*
    uv run python -m generator.score --checkpoint outputs/checkpoints/ddpm/best --samples 100 --seed 1

Each checkpoint appends one row per generated sample to --output (default outputs/metrics/scores.jsonl), tagged with
the checkpoint path and the epoch it was saved after, and prints its accuracy and mean confidence.
"""

import argparse
import json
from pathlib import Path

from generator.classifier import load_classifier
from generator.training import (
    OUTPUT_DIR,
    classifier_records,
    classifier_scores,
    classifier_summary,
    ensure_outside_data,
    load_checkpoint,
)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True, help="one or more checkpoint directories")
    parser.add_argument("--samples", type=int, default=16, help="images generated per digit (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="training uses its --seed, so 0 matches a default run")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "metrics" / "scores.jsonl")
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    ensure_outside_data(args.output)

    classifier = load_classifier()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoint:
        model, diffusion, config = load_checkpoint(checkpoint)
        _, labels, probabilities = classifier_scores(model, diffusion, classifier, args.samples, args.seed)
        tag = {"checkpoint": str(checkpoint), "epoch": config["training"]["epochs_completed"]}
        with args.output.open("a") as file:
            for record in classifier_records(labels, probabilities):
                file.write(json.dumps(tag | record) + "\n")
        summary = classifier_summary(labels, probabilities)
        print(
            f"{checkpoint}: accuracy {summary['classifier_accuracy']:.3f},"
            f" confidence {summary['classifier_confidence']:.3f}"
        )


if __name__ == "__main__":
    main()
