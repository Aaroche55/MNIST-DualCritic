import io
import struct
from dataclasses import dataclass
from hashlib import sha512
from pathlib import Path
from typing import ClassVar, Self
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy import ndimage

# IDX file format, shared by MNIST and EMNIST
IDX_LABELS_MAGIC = 2049
IDX_IMAGES_MAGIC = 2051


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
        self.formatted_images = self.formatted_dir / "images.idx3-ubyte"
        self.formatted_labels = self.formatted_dir / "labels.idx1-ubyte"

    def download(self):
        if self.raw_dir.exists():
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
        assert len(images) == len(labels), f"Dataset sizes do not match: {len(images)} images != {len(labels)} labels"

        write_idx_images(self.formatted_images, images)
        write_idx_labels(self.formatted_labels, labels)

    def load(self) -> Dataset:
        return Dataset(images=read_idx_images(self.formatted_images), labels=read_idx_labels(self.formatted_labels))


class MNIST(SourceDataset):
    name = "mnist"
    url = "https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset"
    sha512 = "b3ec3381298ea83eb56e34fdf84e32b9c6758732707d53b560f23eed662b07f90f406bc1e818d4f66cd2948546c959da6ec2ada44d3d069d26a37bd973d40047"
    raw_files = (
        ("train-images-idx3-ubyte/train-images-idx3-ubyte", "train-labels-idx1-ubyte/train-labels-idx1-ubyte"),
        ("t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"),
    )


def Rotation(source_dataset: Dataset, rng: np.random.Generator | None = None, max_degrees: float = 50.0) -> Dataset:
    rng = rng or np.random.default_rng()
    n = len(source_dataset.images)
    std_degrees = max_degrees / 3  # ~99.7% of draws fall within the limit before clipping
    degrees = np.clip(rng.standard_normal(n) * std_degrees, -max_degrees, max_degrees)
    images = np.empty_like(source_dataset.images)
    for i, (image, angle) in enumerate(zip(source_dataset.images, degrees)):
        ndimage.rotate(image, angle, reshape=False, order=1, mode="constant", cval=0, output=images[i])
    return Dataset(images=images, labels=source_dataset.labels.copy())

def Translation(source_dataset: Dataset, rng: np.random.Generator | None = None, std_pixels: float = 2.0) -> Dataset:
    rng = rng or np.random.default_rng()
    n = len(source_dataset.images)
    # Whole-pixel shifts keep digits sharp; extreme draws may clip the digit at the border
    offsets = np.rint(rng.standard_normal((n, 2)) * std_pixels)
    images = np.empty_like(source_dataset.images)
    for i, (image, offset) in enumerate(zip(source_dataset.images, offsets)):
        ndimage.shift(image, offset, order=0, mode="constant", cval=0, output=images[i])
    return Dataset(images=images, labels=source_dataset.labels.copy())

def Brightness(source_dataset: Dataset, rng: np.random.Generator | None = None, min_scale: float = 0.5, max_scale: float = 1.5) -> Dataset:
    rng = rng or np.random.default_rng()
    n = len(source_dataset.images)
    scale = rng.uniform(min_scale, max_scale, size=(n, 1, 1))
    images = np.clip(np.rint(source_dataset.images * scale), 0, 255).astype(source_dataset.images.dtype)
    return Dataset(images=images, labels=source_dataset.labels.copy())

def get_dataset(source: SourceDataset, *transformations):
    source.download()
    source.format()
    source_dataset = source.load()
    dest_dataset = Dataset.empty(*source_dataset.images.shape[1:])
    dest_dataset.extend(source_dataset)

    rng = np.random.default_rng()
    for transform in transformations:
        dest_dataset.extend(transform(source_dataset, rng))

    return dest_dataset

def main():
    # cli for interacting with datasets
    # use argparse
    # options to download/redownload the dataset
    # generate different translations
    print("Hello, from data.")

    out = get_dataset(MNIST(), Rotation)

    for image_a, image_b in zip(out.images[80000:80010], out.images[10000:10010]):
        for image in (image_a, image_b):
            for x in image:
                for y in x:
                    print([' ','.','"','-','+','o','O','M','#','@'][int(y/256*10)], end="")
                print()

if __name__ == "__main__":
    main()
