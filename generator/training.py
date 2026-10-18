"""Train the class-conditional DDPM on augmented MNIST and track denoising loss on a held-out validation set.

Usage (from the repository root)::

    uv run python -m generator.training --epochs 20 --run-name ddpm
    uv run python -m generator.training --validation 0.2 --split-seed 1 -t rotation -t translation
    uv run python -m generator.training --resume outputs/checkpoints/ddpm/last --epochs 30

The data is built on every run through ``mnist_dualcritic.data`` (see ``load_data``): validation is held out from
the official train partition, then only the remaining images are augmented to make train. The official test
partition is never loaded, so it cannot be used for training or for choosing checkpoints. The data library
downloads MNIST and caches transform outputs under ``data/`` on first use; the generator itself never writes
there. Outputs::

    outputs/checkpoints/<run>/last/   model.safetensors, optimizer.safetensors, config.json (after every epoch)
    outputs/checkpoints/<run>/best/   the same, for the epoch with the lowest validation loss so far
    outputs/metrics/<run>.jsonl       one line per epoch: train and validation loss
"""

import argparse
import json
import math
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from generator.diffusion import DiffusionConfig, GaussianDiffusion
from generator.model import ConditionalUNet, ModelConfig
from mnist_dualcritic.cli import count_or_fraction
from mnist_dualcritic.data import (
    DATASETS,
    TRANSFORMS,
    Brightness,
    Dataset,
    Rotation,
    Subset,
    Transform,
    Translation,
    get_dataset,
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

NORMALISATION = {
    "source": "uint8 greyscale in [0, 255], arrays of shape (N, 28, 28)",
    "model_input": "float32 x / 127.5 - 1 in [-1, 1], NHWC shape (N, 28, 28, 1)",
    "output_to_uint8": "round((clip(x, -1, 1) + 1) * 127.5)",
}

MODEL_FILENAME = "model.safetensors"
OPTIMIZER_FILENAME = "optimizer.safetensors"
CONFIG_FILENAME = "config.json"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 20
    batch_size: int = 128
    eval_batch_size: int = 500
    learning_rate: float = 2e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    run_name: str = "ddpm"
    output_dir: str = str(OUTPUT_DIR)


@dataclass(frozen=True)
class DataConfig:
    """How train and validation are built: which images are held out, and how the rest are augmented."""

    dataset: str = "mnist"
    data_dir: str = str(DATA_DIR)
    validation: int | float = 10_000  # images held out of the train partition: a count or a fraction
    split_seed: int = 0
    stratify: bool = True
    transforms: tuple[Transform, ...] = (Rotation(), Translation(), Brightness())
    include_original: bool = True

    def record(self) -> dict:
        """JSON-ready settings; asdict() would drop each transform's name, which is a class attribute."""
        return asdict(self) | {"transforms": [{"name": t.name, "params": t.params} for t in self.transforms]}


def ensure_outside_data(path: Path):
    """Refuse to write anywhere under data/, which is read-only for the generator."""
    if path.resolve().is_relative_to(DATA_DIR.resolve()):
        raise ValueError(f"Refusing to write under {DATA_DIR}/: {path}")


def load_data(config: DataConfig) -> tuple[Subset, Subset]:
    """(train, validation): validation held out from the official train partition, then train augmented."""
    source = DATASETS[config.dataset](Path(config.data_dir))
    pool, _ = get_dataset(source)  # the official test partition is kept for final evaluation only
    train, validation = pool.split(config.validation, seed=config.split_seed, stratify=config.stratify)
    return train.transform(*config.transforms, include_original=config.include_original), validation


def data_record(train: Subset, validation: Subset) -> dict:
    """Exactly which images and augmentations each side holds, as saved in checkpoint configs."""
    # Round-trip through JSON so a record built now compares equal to one read back from config.json
    return json.loads(json.dumps({"train": train.describe(), "validation": validation.describe()}))


def ensure_same_data(saved: dict, train: Subset, validation: Subset):
    """Refuse to resume on different data, which could put a run's validation images into its training set."""
    if saved.get("data") != data_record(train, validation):
        raise ValueError(
            "This checkpoint was trained on different data; resume with the --validation, --split-seed and"
            " --transform options it was started with (see the data_config section of its config.json)"
        )


def normalise(images: np.ndarray) -> mx.array:
    """uint8 (B, 28, 28) in [0, 255] -> float32 (B, 28, 28, 1) in [-1, 1]."""
    return mx.array(images.astype(np.float32)[..., None] / 127.5 - 1.0)


def denormalise(x: mx.array) -> np.ndarray:
    """float32 (B, 28, 28, 1) -> uint8 (B, 28, 28), clipping to [-1, 1] first."""
    images = (np.clip(np.array(x[..., 0]), -1.0, 1.0) + 1.0) * 127.5
    return np.rint(images).astype(np.uint8)


def batches(dataset: Dataset, batch_size: int, order: np.ndarray | None = None) -> Iterator[tuple[mx.array, mx.array]]:
    """Yield (images (B, 28, 28, 1) float32 in [-1, 1], labels (B,) int32) in `order` (default: file order)."""
    order = np.arange(len(dataset)) if order is None else order
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        yield normalise(dataset.images[index]), mx.array(dataset.labels[index].astype(np.int32))


def derived_seed(seed: int, *stream: int) -> int:
    """A reproducible seed for one (purpose, epoch) stream, independent of how many draws other streams made."""
    return int(np.random.SeedSequence([seed, *stream]).generate_state(1)[0])


def make_optimizer(config: TrainConfig) -> optim.Adam:
    return optim.Adam(learning_rate=config.learning_rate, betas=list(config.adam_betas), eps=config.adam_eps)


def make_train_step(model: ConditionalUNet, diffusion: GaussianDiffusion, optimizer: optim.Optimizer):
    """Return step(images, labels, key) -> loss, which applies one optimiser update to the model."""

    def loss_fn(images, labels, key):
        return diffusion.loss(model, images, labels, key)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(images: mx.array, labels: mx.array, key: mx.array) -> float:
        loss, grads = loss_and_grad(images, labels, key)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        return loss.item()

    return step


def evaluate(model: ConditionalUNet, diffusion: GaussianDiffusion, dataset: Dataset, batch_size: int, seed: int):
    """Mean denoising loss with timesteps and noise fixed by `seed`, so values are comparable across epochs."""
    num_batches = math.ceil(len(dataset) / batch_size)
    keys = mx.random.split(mx.random.key(derived_seed(seed, 2)), num_batches)
    total = 0.0
    for key, (images, labels) in zip(keys, batches(dataset, batch_size), strict=True):
        total += diffusion.loss(model, images, labels, key).item() * images.shape[0]
    return total / len(dataset)


def checkpoint_config(
    train_config: TrainConfig,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    data_config: DataConfig,
    data: dict,
    epoch: int,
    step: int,
    metrics: dict,
) -> dict:
    return {
        "model": asdict(model_config),
        "diffusion": asdict(diffusion_config),
        "optimizer": {
            "name": "Adam",
            "learning_rate": train_config.learning_rate,
            "betas": list(train_config.adam_betas),
            "eps": train_config.adam_eps,
        },
        "training": asdict(train_config) | {"epochs_completed": epoch, "step": step},
        "seed": train_config.seed,
        "normalisation": NORMALISATION,
        "data_config": data_config.record(),
        "data": data,
        "metrics": metrics,
    }


def save_checkpoint(directory: Path, model: ConditionalUNet, optimizer: optim.Optimizer, config: dict):
    """Write weights, optimiser state and config.json, swapping the directory in only once all three exist."""
    directory = Path(directory)
    ensure_outside_data(directory)
    partial = directory.with_name(directory.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    model.save_weights(str(partial / MODEL_FILENAME))
    mx.save_safetensors(str(partial / OPTIMIZER_FILENAME), dict(tree_flatten(optimizer.state)))
    (partial / CONFIG_FILENAME).write_text(json.dumps(config, indent=2) + "\n")
    shutil.rmtree(directory, ignore_errors=True)
    partial.rename(directory)


def load_checkpoint(directory: Path) -> tuple[ConditionalUNet, GaussianDiffusion, dict]:
    """Rebuild the model and diffusion process from a checkpoint directory; returns (model, diffusion, config)."""
    directory = Path(directory)
    config = json.loads((directory / CONFIG_FILENAME).read_text())
    model = ConditionalUNet(ModelConfig(**config["model"]))
    model.load_weights(str(directory / MODEL_FILENAME))
    mx.eval(model.parameters())
    return model, GaussianDiffusion(DiffusionConfig(**config["diffusion"])), config


def load_optimizer_state(directory: Path, optimizer: optim.Optimizer):
    optimizer.state = tree_unflatten(list(mx.load(str(Path(directory) / OPTIMIZER_FILENAME)).items()))


def train(
    config: TrainConfig = TrainConfig(),  # noqa: B008 (frozen dataclasses, safe defaults)
    model_config: ModelConfig = ModelConfig(),  # noqa: B008
    diffusion_config: DiffusionConfig = DiffusionConfig(),  # noqa: B008
    data_config: DataConfig = DataConfig(),  # noqa: B008
    resume: Path | None = None,
) -> Path:
    """Train, saving `last` and `best` checkpoints every epoch; returns the run's checkpoint directory."""
    run_dir = Path(config.output_dir) / "checkpoints" / config.run_name
    metrics_path = Path(config.output_dir) / "metrics" / f"{config.run_name}.jsonl"
    for path in (run_dir, metrics_path):
        ensure_outside_data(path)

    train_subset, validation_subset = load_data(data_config)
    data = data_record(train_subset, validation_subset)

    start_epoch, step, best_loss = 0, 0, math.inf
    if resume is not None:
        model, diffusion, saved = load_checkpoint(resume)
        ensure_same_data(saved, train_subset, validation_subset)
        model_config, diffusion_config = model.config, diffusion.config
        optimizer = make_optimizer(config)
        load_optimizer_state(resume, optimizer)
        start_epoch, step = saved["training"]["epochs_completed"], saved["training"]["step"]
        best_loss = saved["metrics"].get("best_validation_loss", math.inf)
    else:
        mx.random.seed(config.seed)  # parameter initialisation
        model = ConditionalUNet(model_config)
        diffusion = GaussianDiffusion(diffusion_config)
        optimizer = make_optimizer(config)
    mx.eval(model.parameters())

    train_set, validation_set = train_subset.load(), validation_subset.load()
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"train {len(train_set)}, validation {len(validation_set)}, parameters {num_params:,}")

    train_step = make_train_step(model, diffusion, optimizer)
    num_batches = math.ceil(len(train_set) / config.batch_size)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, config.epochs):
        # Each epoch's shuffle and noise come from its own seeded stream, so a resumed run matches an unbroken one
        order = np.random.default_rng(derived_seed(config.seed, 0, epoch)).permutation(len(train_set))
        keys = mx.random.split(mx.random.key(derived_seed(config.seed, 1, epoch)), num_batches)
        total = 0.0
        for i, (key, (images, labels)) in enumerate(
            zip(keys, batches(train_set, config.batch_size, order), strict=True), start=1
        ):
            total += train_step(images, labels, key) * images.shape[0]
            step += 1
            if i % 200 == 0:
                print(
                    f"epoch {epoch + 1} batch {i}/{num_batches} loss {total / min(i * config.batch_size, len(train_set)):.4f}"
                )

        train_loss = total / len(train_set)
        validation_loss = evaluate(model, diffusion, validation_set, config.eval_batch_size, config.seed)
        improved = validation_loss < best_loss
        best_loss = min(best_loss, validation_loss)
        metrics = {
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "best_validation_loss": best_loss,
        }
        with metrics_path.open("a") as file:
            file.write(json.dumps({"epoch": epoch + 1, "step": step} | metrics) + "\n")
        print(
            f"epoch {epoch + 1}: train {train_loss:.4f}, validation {validation_loss:.4f}{' (best)' if improved else ''}"
        )

        saved = checkpoint_config(config, model_config, diffusion_config, data_config, data, epoch + 1, step, metrics)
        save_checkpoint(run_dir / "last", model, optimizer, saved)
        if improved:
            save_checkpoint(run_dir / "best", model, optimizer, saved)
    return run_dir


def main(argv: list[str] | None = None):
    defaults, model_defaults, diffusion_defaults, data_defaults = (
        TrainConfig(),
        ModelConfig(),
        DiffusionConfig(),
        DataConfig(),
    )
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="total epochs (also when resuming)")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--base-channels", type=int, default=model_defaults.base_channels)
    parser.add_argument("--timesteps", type=int, default=diffusion_defaults.num_timesteps)
    parser.add_argument("--resume", type=Path, help="checkpoint directory to continue from, e.g. .../last")
    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", default=data_defaults.data_dir)
    data.add_argument(
        "--validation",
        type=count_or_fraction,
        default=data_defaults.validation,
        help="images held out of the train partition: a count, or a fraction such as 0.2 (default: %(default)s)",
    )
    data.add_argument("--split-seed", type=int, default=data_defaults.split_seed)
    data.add_argument(
        "-t",
        "--transform",
        action="append",
        choices=TRANSFORMS,
        help="augmentation for train, with default parameters; repeat for several"
        f" (default: {', '.join(t.name for t in data_defaults.transforms)})",
    )
    data.add_argument("--no-augment", action="store_true", help="train on the original images only")
    args = parser.parse_args(argv)
    if args.no_augment and args.transform:
        parser.error("--no-augment and --transform cannot be combined")
    if args.no_augment:
        transforms = ()
    elif args.transform:
        transforms = tuple(TRANSFORMS[name]() for name in dict.fromkeys(args.transform))
    else:
        transforms = data_defaults.transforms
    data_config = DataConfig(
        data_dir=args.data_dir, validation=args.validation, split_seed=args.split_seed, transforms=transforms
    )
    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )
    train(
        config,
        ModelConfig(base_channels=args.base_channels),
        DiffusionConfig(num_timesteps=args.timesteps),
        data_config,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
