"""Generate class-conditional MNIST digits from a saved checkpoint and save them as a PNG grid.

Usage (from the repository root)::

    uv run python -m generator.sample --checkpoint outputs/checkpoints/ddpm/best --digit 7
    uv run python -m generator.sample --checkpoint outputs/checkpoints/ddpm/best --digit 3 \\
        --count 64 --seed 1 --output-dir outputs/generated
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from generator.diffusion import GaussianDiffusion
from generator.images import make_grid, write_png
from generator.model import ConditionalUNet
from generator.training import OUTPUT_DIR, denormalise, ensure_outside_data, load_checkpoint


def generate(model: ConditionalUNet, diffusion: GaussianDiffusion, digit: int, count: int, seed: int) -> np.ndarray:
    """`count` samples of `digit` as uint8 (count, H, W); the same seed and checkpoint give the same images."""
    if not 0 <= digit < model.config.num_classes:
        raise ValueError(f"digit must be in [0, {model.config.num_classes}), got {digit}")
    size, channels = model.config.image_size, model.config.image_channels
    labels = mx.full((count,), digit, dtype=mx.int32)
    x = diffusion.sample(model, labels, mx.random.key(seed), image_shape=(size, size, channels))
    return denormalise(x)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint directory, e.g. .../best")
    parser.add_argument("--digit", type=int, required=True, help="digit label to generate, 0-9")
    parser.add_argument("--count", type=int, default=16, help="number of samples in the grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "generated")
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be at least 1")
    ensure_outside_data(args.output_dir)

    model, diffusion, _ = load_checkpoint(args.checkpoint)
    images = generate(model, diffusion, args.digit, args.count, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "-".join(args.checkpoint.resolve().parts[-2:])  # e.g. ddpm-best
    path = args.output_dir / f"{checkpoint_name}_digit{args.digit}_n{args.count}_seed{args.seed}.png"
    write_png(path, make_grid(images))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
