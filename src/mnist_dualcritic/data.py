"""Source datasets, augmentation transforms, and the on-disk cache that composes them.

Library usage::

    from mnist_dualcritic.data import MNIST, Brightness, Rotation, Translation, get_dataset

    source = MNIST()  # files live under ./data/mnist/; MNIST(Path("elsewhere")) changes the root

    # The official partitions. Test stays exactly as published, since pretrained MNIST classifiers (such as the
    # OCR critic) were trained on the official train partition and are only fairly scored on its test partition
    pool, test = get_dataset(source)

    # Split first: hold out a stratified validation set of original images (a count, or a fraction like 0.2).
    # Then augment only the remaining images to make train, so no copy of a validation image is trained on
    train, validation = pool.split(10_000, seed=0)
    train = train.transform(Rotation(), Translation(std_pixels=3))
    # pool.transform(...).split(...) raises: splitting must come before transforming

    # Or k-fold cross-validation: five (train, validation) pairs whose validation sides partition the pool
    for train, validation in pool.kfold(5, seed=0):
        train = train.transform(Rotation())
        ...

    # Subsets are lazy descriptions; nothing is read or generated until load()
    dataset = train.load()
    dataset.images  # (N, 28, 28) uint8: the originals, then one augmented copy per transform
    dataset.labels  # (N,) uint8
    train.describe()                       # JSON-ready provenance: partition, indices hash, transforms
    train.save(source.splits_dir / "train")  # IDX files plus metadata.json, for Dataset.from_directory

    # Different seeds give differently randomised copies of the same transform
    rotations = pool.transform(*(Rotation(seed=s) for s in range(3)), include_original=False)

    source.prepare()                # every original sample, all partitions, downloading and formatting if needed
    source.partitions()             # {"train": range(0, 60000), "test": range(60000, 70000)}
    source.transformed(Rotation())  # a single transform's output over the whole source
    source.cached_transforms()      # {key: metadata} for every cached output

On disk, each dataset gets its own directory::

    data/<dataset>/
        raw/                       downloaded files, each verified by SHA-256
        formatted/                 the partitions merged into one IDX image/label pair, with metadata.json
                                   recording where each partition lies
        transforms/<name>-<hash>/  one cached transform output: IDX images, labels and metadata.json
        splits/<name>/             subsets written by Subset.save() or ``data split``

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

Transforms are applied to the whole source and cached once; subsets take rows from that output. So a transform's
output for one sample must not depend on any other sample.

Adding a dataset means subclassing ``SourceDataset`` with ``name``, ``url``, ``raw_files`` (one file pair per
partition in ``PARTITIONS``) and ``checksums``; override ``read_raw()`` if the raw files need more than reading
(EMNIST's images are stored transposed).

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
from typing import ClassVar, NamedTuple, Self
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

# Every source ships these official partitions. The test partition is what pretrained MNIST classifiers are
# scored on, so it is kept exactly as published and never mixed into training or validation.
PARTITIONS = ("train", "test")

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


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


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


def _save_with_metadata(directory: Path, dataset: Dataset, metadata: dict):
    """Write a dataset and its metadata.json, describing the files as well as how they were made.

    The output is built beside `directory` and swapped in, so an interrupted run never leaves a directory that
    looks complete (e.g. old metadata next to half-written images).
    """
    partial = directory.with_name(directory.name + PARTIAL_SUFFIX)
    shutil.rmtree(partial, ignore_errors=True)
    dataset.save(partial)
    metadata = {
        **metadata,
        "samples": len(dataset),
        "image_shape": list(dataset.images.shape[1:]),
        "label_counts": np.bincount(dataset.labels).tolist(),
        "sha256": {filename: _sha256(partial / filename) for filename in (IMAGES_FILENAME, LABELS_FILENAME)},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (partial / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.rmtree(directory, ignore_errors=True)
    partial.rename(directory)


@dataclass(frozen=True)
class Transform:
    """A deterministic augmentation. Its fields are its parameters, and together they identify its cached output.

    Bump `version` whenever the transform's algorithm changes so stale cached outputs are regenerated.

    A transform is applied to the whole source once and its output cached; subsets then take the rows they need.
    So sample i's output must depend only on sample i and the i-th draws from `rng`, never on other samples.
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
    # partition -> (images, labels) file pair relative to the raw directory, one entry per name in PARTITIONS
    raw_files: ClassVar[dict[str, tuple[str, str]]]
    # SHA-256 of every file in raw_files, so verification ignores unrelated files in the raw directory
    checksums: ClassVar[dict[str, str]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "raw_files" in cls.__dict__ and set(cls.raw_files) != set(PARTITIONS):
            raise ValueError(f"{cls.__qualname__}.raw_files must have exactly the partitions {PARTITIONS}")
        _register(DATASETS, cls)

    def __init__(self, data_dir: Path = Path("./data")):
        self.raw_dir = data_dir / self.name / "raw"
        self.formatted_dir = data_dir / self.name / "formatted"
        self.transforms_dir = data_dir / self.name / "transforms"
        self.splits_dir = data_dir / self.name / "splits"

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

    def read_raw(self, partition: str) -> Dataset:
        """One partition straight from the raw files. Override if they need more than reading (e.g. transposing)."""
        images, labels = self.raw_files[partition]
        return Dataset(images=read_idx_images(self.raw_dir / images), labels=read_idx_labels(self.raw_dir / labels))

    def format(self):
        """Merge the partitions, in PARTITIONS order, into one dataset that records where each one lies."""
        parts = [self.read_raw(partition) for partition in PARTITIONS]
        bounds = np.cumsum([0, *map(len, parts)]).tolist()
        metadata = {
            "source": self.name,
            "partitions": {name: bounds[i : i + 2] for i, name in enumerate(PARTITIONS)},
        }
        _save_with_metadata(self.formatted_dir, Dataset.concatenate(parts), metadata)

    def load(self) -> Dataset:
        return Dataset.from_directory(self.formatted_dir)

    def ensure_formatted(self, force_download: bool = False):
        """Download and format if needed. Formatted data without metadata predates partitions and is rebuilt."""
        if force_download or not (self.formatted_dir / METADATA_FILENAME).exists():
            self.download(force=force_download)
            self.format()

    def prepare(self, force_download: bool = False) -> Dataset:
        """Download and format if needed, then load every sample, all partitions included."""
        self.ensure_formatted(force_download)
        return self.load()

    def partitions(self) -> dict[str, range]:
        """Where each official partition lies in the formatted dataset."""
        self.ensure_formatted()
        metadata = json.loads((self.formatted_dir / METADATA_FILENAME).read_text())
        return {name: range(*bounds) for name, bounds in metadata["partitions"].items()}

    def labels(self) -> np.ndarray:
        """Every sample's label, without reading the images."""
        self.ensure_formatted()
        return read_idx_labels(self.formatted_dir / LABELS_FILENAME)

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
        }
        _save_with_metadata(directory, dataset, metadata)
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
    raw_files: ClassVar[dict[str, tuple[str, str]]] = {
        "train": (
            "train-images-idx3-ubyte/train-images-idx3-ubyte",
            "train-labels-idx1-ubyte/train-labels-idx1-ubyte",
        ),
        "test": ("t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"),
    }
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


class Subset:
    """Some of a source's samples, plus the augmented copies to add, read from disk only when load() is called.

    Subsets are cheap descriptions, so splitting and transforming them costs nothing until you load. Split first,
    then transform: split() and kfold() refuse a subset that already has transforms, so validation is always
    held out from the original images and no augmented copy of a validation image can reach training.
    """

    def __init__(
        self,
        source: SourceDataset,
        partition: str,
        indices,
        transforms: tuple[Transform, ...] = (),
        include_original: bool = True,
    ):
        self.source = source
        self.partition = partition
        # Rows of the formatted source; sorted so a subset loads in source order however it was drawn
        self.indices = np.sort(np.asarray(indices, dtype=np.int64))
        self.indices.flags.writeable = False
        self.transforms = tuple(transforms)
        self.include_original = include_original

    def __len__(self) -> int:
        return len(self.indices) * (self.include_original + len(self.transforms))

    def __repr__(self) -> str:
        blocks = (["original"] if self.include_original else []) + [t.key for t in self.transforms]
        return (
            f"Subset({self.source.name}/{self.partition}, {len(self.indices)} images x [{', '.join(blocks)}]"
            f" = {len(self)} samples)"
        )

    def _require_untransformed(self, operation: str):
        if self.transforms or not self.include_original:
            raise ValueError(
                f"Cannot {operation} a transformed subset: {operation} first, then transform the training side,"
                " so validation is held out from the original images only"
            )

    def transform(self, *transforms: Transform, include_original: bool = True) -> "Subset":
        """Add one augmented copy of every image per transform. include_original=False drops the unmodified images."""
        return Subset(
            self.source,
            self.partition,
            self.indices,
            self.transforms + transforms,
            self.include_original and include_original,
        )

    def _order(self, seed: int, stratify: bool) -> np.ndarray:
        """A seeded shuffle of positions in `indices`; stratified, any prefix has the subset's class balance."""
        order = np.random.default_rng(seed).permutation(len(self.indices))
        if not stratify:
            return order
        labels = self.source.labels()[self.indices[order]]
        # Spread each class evenly through the order: key each image by its rank within its class as a fraction
        # of the class size, so every contiguous run of the order takes the same share of each class
        position = np.empty(len(order))
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            position[members] = (np.arange(len(members)) + 0.5) / len(members)
        return order[np.argsort(position, kind="stable")]

    def split(self, validation: float | int, *, seed: int = 0, stratify: bool = True) -> tuple["Subset", "Subset"]:
        """Hold out `validation` images (a count, or a fraction of the images) and return (train, validation).

        Both sides are original images; transform the training side afterwards. Classes keep their proportions
        on both sides unless stratify=False.
        """
        self._require_untransformed("split")
        n = len(self.indices)
        if isinstance(validation, float):
            if not 0 < validation < 1:
                raise ValueError(f"A validation fraction must be between 0 and 1, got {validation}")
            validation = round(validation * n)
        if not 0 < validation < n:
            raise ValueError(f"Cannot hold out {validation} of {n} images: both sides need at least one")
        order = self.indices[self._order(seed, stratify)]
        return Subset(self.source, self.partition, order[validation:]), Subset(
            self.source, self.partition, order[:validation]
        )

    def kfold(self, k: int, *, seed: int = 0, stratify: bool = True) -> list[tuple["Subset", "Subset"]]:
        """k (train, validation) pairs whose validation sides partition the images, each of near-equal size.

        Like split(), both sides are original images; transform each training side afterwards.
        """
        self._require_untransformed("kfold")
        n = len(self.indices)
        if not 2 <= k <= n:
            raise ValueError(f"k must be between 2 and the number of images ({n}), got {k}")
        # Contiguous runs of the stratified order, so each fold is balanced like a split's validation side
        fold = np.empty(n, dtype=np.int64)
        fold[self._order(seed, stratify)] = np.arange(n) * k // n
        return [
            (
                Subset(self.source, self.partition, self.indices[fold != i]),
                Subset(self.source, self.partition, self.indices[fold == i]),
            )
            for i in range(k)
        ]

    def load(self) -> Dataset:
        """The selected originals, if included, followed by each transform's copies, generating any not cached."""
        original = self.source.prepare()
        parts = [original[self.indices]] if self.include_original else []
        parts += [self.source.transformed(transform)[self.indices] for transform in self.transforms]
        return Dataset.concatenate(parts) if parts else original[self.indices[:0]]

    def describe(self) -> dict:
        """Exactly which samples and augmentations this subset holds, as JSON-ready provenance."""
        return {
            "source": self.source.name,
            "partition": self.partition,
            "images": len(self.indices),
            "indices_sha256": sha256(self.indices.astype("<i8").tobytes()).hexdigest(),
            "include_original": self.include_original,
            "transforms": [
                {"key": t.key, "transform": t.name, "version": t.version, "params": t.params} for t in self.transforms
            ],
        }

    def save(self, directory: Path) -> Dataset:
        """Load and write the subset as IDX files with metadata.json, readable by Dataset.from_directory()."""
        dataset = self.load()
        _save_with_metadata(directory, dataset, self.describe())
        return dataset


class Partitions(NamedTuple):
    train: Subset
    test: Subset


def get_dataset(source: SourceDataset) -> Partitions:
    """The source's official (train, test) partitions, downloading and formatting it first if needed.

    Split validation sets out of train. Keep test as it is: pretrained classifiers were trained on the official
    train partition, so only the official test partition gives them a fair score.
    """
    return Partitions(**{name: Subset(source, name, rows) for name, rows in source.partitions().items()})
