"""Source datasets, augmentation transforms, and the on-disk cache that composes them.

Library usage::

    from mnist_dualcritic.data import MNIST, Brightness, Rotation, Translation, get_dataset

    source = MNIST()  # files live under ./data/mnist/; MNIST(Path("elsewhere")) changes the root

    # The original samples plus one augmented copy per transform, each generated and cached on first use
    dataset = get_dataset(source, Rotation(), Translation(std_pixels=3), Brightness(seed=1))
    dataset.images  # (N, 28, 28) uint8
    dataset.labels  # (N,) uint8

    # Different seeds give differently randomised copies of the same transform
    rotations = get_dataset(source, *(Rotation(seed=s) for s in range(3)), include_original=False)

    source.prepare()                # just the original data, downloading and formatting it if needed
    source.transformed(Rotation())  # a single transform's output
    source.cached_transforms()      # {key: metadata} for every cached output

On disk, each dataset gets its own directory::

    data/<dataset>/
        raw/                       downloaded files, each verified by SHA-256
        formatted/                 the train and test splits merged into one IDX image/label pair
        transforms/<name>-<hash>/  one cached transform output: IDX images, labels and metadata.json

Transforms are frozen dataclasses whose fields, including ``seed``, are their parameters. Each one is
deterministic, so its output is cached under a key hashed from its name, ``version`` and parameters, and
``metadata.json`` records exactly how that output was made. Bump ``version`` after changing a transform's
algorithm so outputs cached by the old code are regenerated instead of reused.

Adding a transform::

    @dataclass(frozen=True)
    class Invert(Transform):
        name = "invert"        # registers it in TRANSFORMS, and so in the CLI's --transform choices
        amount: float = 1.0    # each field is a parameter, part of the cache key, and a CLI flag (--amount)

        def apply(self, dataset: Dataset, rng: np.random.Generator) -> Dataset:
            images = np.rint(np.abs(dataset.images - self.amount * 255)).astype(np.uint8)
            return Dataset(images=images, labels=dataset.labels.copy())

Adding a dataset means subclassing ``SourceDataset`` with ``name``, ``url``, ``raw_files`` and ``checksums``;
override ``format()`` if the raw files need more than concatenating (EMNIST's images are stored transposed).

For the command line, see ``mnist_dualcritic.cli``.
"""

import io
import json
import shutil
import struct
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from hashlib import file_digest, sha256
from pathlib import Path
from typing import ClassVar, Self
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy import ndimage

# IDX file format, shared by MNIST and EMNIST
IDX_LABELS_MAGIC = 2049
IDX_IMAGES_MAGIC = 2051

# Every stored dataset directory (formatted source or cached transform) uses these file names
IMAGES_FILENAME = "images.idx3-ubyte"
LABELS_FILENAME = "labels.idx1-ubyte"
METADATA_FILENAME = "metadata.json"
PARTIAL_SUFFIX = ".partial"

# Name -> class lookups used by the CLI; each subclass registers itself when defined, wherever it lives
DATASETS: dict[str, type["SourceDataset"]] = {}
TRANSFORMS: dict[str, type["Transform"]] = {}


def _register(registry: dict[str, type], cls: type):
    if "name" not in cls.__dict__:
        return  # intermediate base classes without their own name aren't selectable
    if cls.name in registry:
        raise ValueError(f"{cls.name!r} is already registered by {registry[cls.name].__qualname__}")
    registry[cls.name] = cls


def _read_idx(path: Path, magic: int, ndim: int, kind: str) -> np.ndarray:
    """Read an unsigned-byte IDX file with `ndim` dimensions, validating it against its header."""
    header_size = 4 * (1 + ndim)
    with path.open("rb") as file:
        header = file.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"{path}: file is too short to contain an IDX header")
        found_magic, *shape = struct.unpack(f">{1 + ndim}I", header)
        if found_magic != magic:
            raise ValueError(f"{path}: Magic number mismatch, expected {magic}, got {found_magic}")
        body = file.read()
    item_size = int(np.prod(shape[1:]))
    if len(body) != shape[0] * item_size:
        raise ValueError(f"{path}: header says {shape[0]} {kind}, found {len(body) / item_size:g}")
    # Copy so callers get a writable array rather than a read-only view of the bytes
    return np.frombuffer(body, dtype=np.uint8).reshape(shape).copy()


def read_idx_labels(path: Path) -> np.ndarray:
    return _read_idx(path, IDX_LABELS_MAGIC, ndim=1, kind="labels")


def read_idx_images(path: Path) -> np.ndarray:
    return _read_idx(path, IDX_IMAGES_MAGIC, ndim=3, kind="images")


def _require_uint8(name: str, array: np.ndarray):
    # Casting would silently wrap out-of-range values (e.g. 300 -> 44), so make callers convert explicitly
    if array.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8 to write as IDX, got {array.dtype}")


def write_idx_labels(path: Path, labels: np.ndarray):
    _require_uint8("labels", labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(struct.pack(">II", IDX_LABELS_MAGIC, len(labels)))
        file.write(labels.tobytes())


def write_idx_images(path: Path, images: np.ndarray):
    _require_uint8("images", images)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(struct.pack(">IIII", IDX_IMAGES_MAGIC, *images.shape))
        file.write(images.tobytes())


@dataclass
class Dataset:
    images: np.ndarray  # (N, rows, cols)
    labels: np.ndarray  # (N,)

    def __post_init__(self):
        self.images = np.asarray(self.images)
        self.labels = np.asarray(self.labels)
        if self.images.ndim != 3:
            raise ValueError(f"images must have shape (N, rows, cols), got {self.images.shape}")
        if self.labels.ndim != 1:
            raise ValueError(f"labels must have shape (N,), got {self.labels.shape}")
        if len(self.images) != len(self.labels):
            raise ValueError(f"Dataset sizes do not match: {len(self.images)} images != {len(self.labels)} labels")

    @classmethod
    def empty(cls, rows: int = 28, cols: int = 28) -> Self:
        return cls(images=np.empty((0, rows, cols), dtype=np.uint8), labels=np.empty((0,), dtype=np.uint8))

    @classmethod
    def concatenate(cls, datasets: list[Self]) -> Self:
        """Join datasets in one copy (repeated extend() re-copies everything joined so far)."""
        return cls(
            images=np.concatenate([d.images for d in datasets]), labels=np.concatenate([d.labels for d in datasets])
        )

    def extend(self, other_dataset: Self):
        self.images = np.concatenate((self.images, other_dataset.images), axis=0)
        self.labels = np.concatenate((self.labels, other_dataset.labels), axis=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index) -> Self:
        if isinstance(index, (int, np.integer)):
            index = [index]  # keep the leading N axis so a single item is still a Dataset
        return Dataset(images=self.images[index], labels=self.labels[index])

    def save(self, directory: Path):
        write_idx_images(directory / IMAGES_FILENAME, self.images)
        write_idx_labels(directory / LABELS_FILENAME, self.labels)

    @classmethod
    def from_directory(cls, directory: Path) -> Self:
        return cls(
            images=read_idx_images(directory / IMAGES_FILENAME), labels=read_idx_labels(directory / LABELS_FILENAME)
        )


@dataclass(frozen=True)
class Transform:
    """A deterministic augmentation. Its fields are its parameters, and together they identify its cached output.

    Bump `version` whenever the transform's algorithm changes so stale cached outputs are regenerated.
    """

    name: ClassVar[str]
    version: ClassVar[int] = 1
    seed: int = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register(TRANSFORMS, cls)

    def __post_init__(self):
        # Normalise ints passed for float parameters so Rotation(max_degrees=50) and 50.0 share a cache entry
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type in (float, "float") and isinstance(value, int) and not isinstance(value, bool):
                object.__setattr__(self, f.name, float(value))

    def __call__(self, dataset: Dataset) -> Dataset:
        return self.apply(dataset, np.random.default_rng(self.seed))

    def apply(self, dataset: Dataset, rng: np.random.Generator) -> Dataset:
        raise NotImplementedError

    @property
    def params(self) -> dict:
        return asdict(self)

    @property
    def key(self) -> str:
        payload = json.dumps({"transform": self.name, "version": self.version, "params": self.params}, sort_keys=True)
        return f"{self.name}-{sha256(payload.encode()).hexdigest()[:12]}"


class SourceDataset:
    """A downloadable IDX dataset. Subclasses describe where it comes from and which raw files it contains."""

    name: ClassVar[str]
    url: ClassVar[str]
    # (images, labels) file pairs relative to the raw directory, merged in order by format()
    raw_files: ClassVar[tuple[tuple[str, str], ...]]
    # SHA-256 of every file in raw_files, so verification ignores unrelated files in the raw directory
    checksums: ClassVar[dict[str, str]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register(DATASETS, cls)

    def __init__(self, data_dir: Path = Path("./data")):
        self.raw_dir = data_dir / self.name / "raw"
        self.formatted_dir = data_dir / self.name / "formatted"
        self.transforms_dir = data_dir / self.name / "transforms"

    def invalid_raw_files(self) -> list[str]:
        """Raw files that are missing or don't match their expected checksum."""
        invalid = []
        for relative_path, expected in self.checksums.items():
            path = self.raw_dir / relative_path
            if not path.is_file():
                invalid.append(relative_path)
                continue
            with path.open("rb") as file:
                if file_digest(file, "sha256").hexdigest() != expected:
                    invalid.append(relative_path)
        return invalid

    def download(self, force: bool = False):
        if not force:
            invalid = self.invalid_raw_files()
            if not invalid:
                print("Raw files present and checksums match, skipping download.")
                return
            print(f"Missing or mismatched raw files ({', '.join(invalid)}), downloading dataset.")

        with urlopen(self.url, timeout=30) as resp:
            data = resp.read()

        with ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(self.raw_dir)

        if invalid := self.invalid_raw_files():
            raise RuntimeError(f"Downloaded {self.name} files failed checksum verification: {', '.join(invalid)}")

    def format(self):
        images = np.concatenate([read_idx_images(self.raw_dir / images) for images, _ in self.raw_files])
        labels = np.concatenate([read_idx_labels(self.raw_dir / labels) for _, labels in self.raw_files])
        Dataset(images=images, labels=labels).save(self.formatted_dir)

    def load(self) -> Dataset:
        return Dataset.from_directory(self.formatted_dir)

    def prepare(self, force_download: bool = False) -> Dataset:
        """Download and format if needed, then load."""
        formatted = (self.formatted_dir / IMAGES_FILENAME, self.formatted_dir / LABELS_FILENAME)
        if force_download or not all(path.exists() for path in formatted):
            self.download(force=force_download)
            self.format()
        return self.load()

    def is_cached(self, transform: Transform) -> bool:
        # A directory without metadata is an interrupted run and is never treated as cached
        return (self.transforms_dir / transform.key / METADATA_FILENAME).exists()

    def transformed(self, transform: Transform, regenerate: bool = False) -> Dataset:
        """Load the transform's cached output, generating and caching it first if needed."""
        directory = self.transforms_dir / transform.key
        if self.is_cached(transform) and not regenerate:
            return Dataset.from_directory(directory)

        dataset = transform(self.prepare())
        metadata = {
            "key": transform.key,
            "source": self.name,
            "transform": transform.name,
            "version": transform.version,
            "params": transform.params,
            "samples": len(dataset),
            "image_shape": list(dataset.images.shape[1:]),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        # Build the output beside the cache and swap it in, so an interrupted run never leaves a
        # directory that looks complete (e.g. old metadata next to half-written images)
        partial = directory.with_name(directory.name + PARTIAL_SUFFIX)
        shutil.rmtree(partial, ignore_errors=True)
        dataset.save(partial)
        (partial / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.rmtree(directory, ignore_errors=True)
        partial.rename(directory)
        return dataset

    def cached_transforms(self) -> dict[str, dict]:
        """Metadata for every cached transform output, keyed by directory name."""
        return {
            path.parent.name: json.loads(path.read_text())
            for path in sorted(self.transforms_dir.glob(f"*/{METADATA_FILENAME}"))
            if not path.parent.name.endswith(PARTIAL_SUFFIX)
        }


class MNIST(SourceDataset):
    name = "mnist"
    url = "https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset"
    raw_files = (
        ("train-images-idx3-ubyte/train-images-idx3-ubyte", "train-labels-idx1-ubyte/train-labels-idx1-ubyte"),
        ("t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"),
    )
    checksums: ClassVar[dict[str, str]] = {
        "train-images-idx3-ubyte/train-images-idx3-ubyte": "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
        "train-labels-idx1-ubyte/train-labels-idx1-ubyte": "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
        "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte": "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
        "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte": "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
    }


@dataclass(frozen=True)
class Rotation(Transform):
    name = "rotation"
    max_degrees: float = 50.0

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.max_degrees <= 180:
            raise ValueError(f"max_degrees must be between 0 and 180, got {self.max_degrees}")

    def apply(self, dataset: Dataset, rng: np.random.Generator) -> Dataset:
        std_degrees = self.max_degrees / 3  # ~99.7% of draws fall within the limit before clipping
        degrees = np.clip(rng.standard_normal(len(dataset)) * std_degrees, -self.max_degrees, self.max_degrees)
        images = np.empty_like(dataset.images)
        for i, (image, angle) in enumerate(zip(dataset.images, degrees, strict=True)):
            ndimage.rotate(image, angle, reshape=False, order=1, mode="constant", cval=0, output=images[i])
        return Dataset(images=images, labels=dataset.labels.copy())


@dataclass(frozen=True)
class Translation(Transform):
    name = "translation"
    std_pixels: float = 2.0

    def __post_init__(self):
        super().__post_init__()
        if self.std_pixels < 0:
            raise ValueError(f"std_pixels must be non-negative, got {self.std_pixels}")

    def apply(self, dataset: Dataset, rng: np.random.Generator) -> Dataset:
        # Whole-pixel shifts keep digits sharp; extreme draws may clip the digit at the border
        offsets = np.rint(rng.standard_normal((len(dataset), 2)) * self.std_pixels)
        images = np.empty_like(dataset.images)
        for i, (image, offset) in enumerate(zip(dataset.images, offsets, strict=True)):
            ndimage.shift(image, offset, order=0, mode="constant", cval=0, output=images[i])
        return Dataset(images=images, labels=dataset.labels.copy())


@dataclass(frozen=True)
class Brightness(Transform):
    name = "brightness"
    min_scale: float = 0.5
    max_scale: float = 1.5

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.min_scale <= self.max_scale:
            raise ValueError(f"Need 0 <= min_scale <= max_scale, got {self.min_scale} and {self.max_scale}")

    def apply(self, dataset: Dataset, rng: np.random.Generator) -> Dataset:
        scale = rng.uniform(self.min_scale, self.max_scale, size=(len(dataset), 1, 1))
        images = np.clip(np.rint(dataset.images * scale), 0, 255).astype(dataset.images.dtype)
        return Dataset(images=images, labels=dataset.labels.copy())


def get_dataset(source: SourceDataset, *transforms: Transform, include_original: bool = True) -> Dataset:
    """Compose the source dataset with each transform's output, generating any that aren't cached yet."""
    original = source.prepare()
    parts = [original] if include_original else []
    parts += [source.transformed(transform) for transform in transforms]
    if not parts:
        return Dataset.empty(*original.images.shape[1:])
    return Dataset.concatenate(parts)
