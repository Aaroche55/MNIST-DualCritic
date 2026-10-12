"""Build the train/validation/test splits and save them under ``data/splits/``.

Usage::

    uv run python scripts/make_splits.py

The 70,000 formatted MNIST samples are sliced in file order into train [0, 42000), test [42000, 56000) and
validation [56000, 70000), before any augmentation, so no augmented copy of a validation or test image can
reach training. The training split is the 42,000 unmodified
images followed by one augmented copy per transform (168,000 samples). Each split directory holds IDX image
and label files plus ``metadata.json``, which records how the split was made and SHA-256 hashes of its files.
"""

import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import file_digest
from pathlib import Path

import numpy as np

from mnist_dualcritic.data import (
    IMAGES_FILENAME,
    LABELS_FILENAME,
    METADATA_FILENAME,
    MNIST,
    Dataset,
    Rotation,
    Translation,
    Brightness
)

SPLITS_DIR = Path("data/splits")
SEED = 42
TRAIN_END = 42000
TEST_END = 56000
TRANSFORMS = [Rotation(), Translation(), Brightness()]


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def save_split(name: str, dataset: Dataset, metadata: dict):
    directory = SPLITS_DIR / name
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    dataset.save(directory)
    metadata |= {
        "split": name,
        "samples": len(dataset),
        "label_counts": np.bincount(dataset.labels, minlength=10).tolist(),
        "sha256": {filename: sha256(directory / filename) for filename in (IMAGES_FILENAME, LABELS_FILENAME)},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (directory / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{name}: {len(dataset)} samples -> {directory}")


def main():
    source = MNIST()
    dataset = source.prepare()
    train = dataset[np.arange(0, TRAIN_END)]
    test = dataset[np.arange(TRAIN_END, TEST_END)]
    validation = dataset[np.arange(TEST_END, len(dataset))]

    rng = np.random.default_rng(SEED)
    parts = [transform.apply(train, rng) for transform in TRANSFORMS]
    new_train = Dataset.concatenate([train, *parts])

    common = {"source": source.name, "source_samples": len(dataset)}
    save_split(
        "train",
        new_train,
        common
        | {
            "source_slice": [0, TRAIN_END],
            "seed": SEED,
            "blocks": ["original"] + [{"transform": t.name, "params": asdict(t)} for t in TRANSFORMS],
        },
    )
    save_split("validation", validation, common | {"source_slice": [TEST_END, len(dataset)]})
    save_split("test", test, common | {"source_slice": [TRAIN_END, TEST_END]})


if __name__ == "__main__":
    main()
