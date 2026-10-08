import io
import json
import struct
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256, sha512
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


def read_idx_labels(path: Path) -> np.ndarray:
    with path.open('rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != IDX_LABELS_MAGIC:
            raise ValueError(f"Magic number mismatch, expected {IDX_LABELS_MAGIC}, got {magic}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    if len(labels) != size:
        raise ValueError(f"{path}: header says {size} labels, found {len(labels)}")
    return labels.copy()

def read_idx_images(path: Path) -> np.ndarray:
    with path.open('rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != IDX_IMAGES_MAGIC:
            raise ValueError(f"Magic number mismatch, expected {IDX_IMAGES_MAGIC}, got {magic}")
        images = np.frombuffer(file.read(), dtype=np.uint8).reshape(-1, rows, cols)
    if len(images) != size:
        raise ValueError(f"{path}: header says {size} images, found {len(images)}")
    return images.copy()

def write_idx_labels(path: Path, labels: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as file:
        file.write(struct.pack(">II", IDX_LABELS_MAGIC, len(labels)))
        file.write(labels.astype(np.uint8).tobytes())

def write_idx_images(path: Path, images: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as file:
        file.write(struct.pack(">IIII", IDX_IMAGES_MAGIC, *images.shape))
        file.write(images.astype(np.uint8).tobytes())


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
        return cls(images=read_idx_images(directory / IMAGES_FILENAME), labels=read_idx_labels(directory / LABELS_FILENAME))


@dataclass(frozen=True)
class Transform:
    """A deterministic augmentation. Its fields are its parameters, and together they identify its cached output.

    Bump `version` whenever the transform's algorithm changes so stale cached outputs are regenerated.
    """

    name: ClassVar[str]
    version: ClassVar[int] = 1
    seed: int = 0

    def __post_init__(self):
        # Normalise ints passed for float parameters so Rotation(max_degrees=50) and 50.0 share a cache entry
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is float and isinstance(value, int) and not isinstance(value, bool):
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
    sha512: ClassVar[str]
    # (images, labels) file pairs relative to the raw directory, merged in order by format()
    raw_files: ClassVar[tuple[tuple[str, str], ...]]

    def __init__(self, data_dir: Path = Path("./data")):
        self.raw_dir = data_dir / self.name / "raw"
        self.formatted_dir = data_dir / self.name / "formatted"
        self.transforms_dir = data_dir / self.name / "transforms"

    def download(self, force: bool = False):
        if self.raw_dir.exists() and not force:
            fileData = b""
            for file in self.raw_dir.rglob("*"):
                if file.is_dir():
                    continue
                fileData += file.read_bytes()

            chksum = sha512(fileData).hexdigest()
            print(f"SHA512 checksum is: {chksum}")
            if chksum == self.sha512:
                print("Checksum matched, skipping download.")
                return
            else:
                print("Checksum does not match, downloading dataset.")

        with urlopen(self.url, timeout=30) as resp:
            data = resp.read()

        with ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(self.raw_dir)

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
        dataset.save(directory)
        metadata = {
            "source": self.name,
            "transform": transform.name,
            "version": transform.version,
            "params": transform.params,
            "samples": len(dataset),
            "image_shape": list(dataset.images.shape[1:]),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        # Written last so is_cached() only sees complete outputs
        (directory / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
        return dataset

    def cached_transforms(self) -> dict[str, dict]:
        """Metadata for every cached transform output, keyed by directory name."""
        return {
            path.parent.name: json.loads(path.read_text())
            for path in sorted(self.transforms_dir.glob(f"*/{METADATA_FILENAME}"))
        }


class MNIST(SourceDataset):
    name = "mnist"
    url = "https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset"
    sha512 = "b3ec3381298ea83eb56e34fdf84e32b9c6758732707d53b560f23eed662b07f90f406bc1e818d4f66cd2948546c959da6ec2ada44d3d069d26a37bd973d40047"
    raw_files = (
        ("train-images-idx3-ubyte/train-images-idx3-ubyte", "train-labels-idx1-ubyte/train-labels-idx1-ubyte"),
        ("t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"),
    )


@dataclass(frozen=True)
class Rotation(Transform):
    name = "rotation"
    max_degrees: float = 50.0

    def apply(self, dataset: Dataset, rng: np.random.Generator) -> Dataset:
        std_degrees = self.max_degrees / 3  # ~99.7% of draws fall within the limit before clipping
        degrees = np.clip(rng.standard_normal(len(dataset)) * std_degrees, -self.max_degrees, self.max_degrees)
        images = np.empty_like(dataset.images)
        for i, (image, angle) in enumerate(zip(dataset.images, degrees)):
            ndimage.rotate(image, angle, reshape=False, order=1, mode="constant", cval=0, output=images[i])
        return Dataset(images=images, labels=dataset.labels.copy())

@dataclass(frozen=True)
class Translation(Transform):
    name = "translation"
    std_pixels: float = 2.0

    def apply(self, dataset: Dataset, rng: np.random.Generator) -> Dataset:
        # Whole-pixel shifts keep digits sharp; extreme draws may clip the digit at the border
        offsets = np.rint(rng.standard_normal((len(dataset), 2)) * self.std_pixels)
        images = np.empty_like(dataset.images)
        for i, (image, offset) in enumerate(zip(dataset.images, offsets)):
            ndimage.shift(image, offset, order=0, mode="constant", cval=0, output=images[i])
        return Dataset(images=images, labels=dataset.labels.copy())

@dataclass(frozen=True)
class Brightness(Transform):
    name = "brightness"
    min_scale: float = 0.5
    max_scale: float = 1.5

    def apply(self, dataset: Dataset, rng: np.random.Generator) -> Dataset:
        scale = rng.uniform(self.min_scale, self.max_scale, size=(len(dataset), 1, 1))
        images = np.clip(np.rint(dataset.images * scale), 0, 255).astype(dataset.images.dtype)
        return Dataset(images=images, labels=dataset.labels.copy())

# Lookup tables used by the CLI; subclasses register themselves by name
DATASETS: dict[str, type[SourceDataset]] = {cls.name: cls for cls in SourceDataset.__subclasses__()}
TRANSFORMS: dict[str, type[Transform]] = {cls.name: cls for cls in Transform.__subclasses__()}

def get_dataset(source: SourceDataset, *transforms: Transform, include_original: bool = True) -> Dataset:
    """Compose the source dataset with each transform's output, generating any that aren't cached yet."""
    original = source.prepare()
    dataset = Dataset.empty(*original.images.shape[1:])
    if include_original:
        dataset.extend(original)
    for transform in transforms:
        dataset.extend(source.transformed(transform))
    return dataset
