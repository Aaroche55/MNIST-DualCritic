import struct
import io
import numpy as np
from scipy import ndimage
from array import array
from dataclasses import dataclass
from enum import Enum, member
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlopen
from hashlib import sha512
from typing import Self

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset"
DATASET_SHA512 = "b3ec3381298ea83eb56e34fdf84e32b9c6758732707d53b560f23eed662b07f90f406bc1e818d4f66cd2948546c959da6ec2ada44d3d069d26a37bd973d40047"
DATA_DIR = Path("./data")
DATA_DIR_RAW = DATA_DIR / "raw"
DATA_DIR_FORMATTED = DATA_DIR / "formatted"
DATA_DIR_FORMATTED_LABELS = DATA_DIR_FORMATTED / "labels.idx1-ubyte"
DATA_DIR_FORMATTED_IMAGES = DATA_DIR_FORMATTED / "images.idx3-ubyte"

# DATA_FORMATTED = DATA_DIR / "formatted"
# DATA_TRANSLATIONS = DATA_DIR / "translations"

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

class MNIST:
    def download():
        if DATA_DIR_RAW.exists():
            fileData = b""
            for file in DATA_DIR_RAW.rglob("*"):
                if file.is_dir():
                    continue
                fileData += file.read_bytes()

            chksum = sha512(fileData).hexdigest()
            print(f"SHA512 checksum is: {chksum}")
            if chksum == DATASET_SHA512:
                print("Checksum matched, skipping download.")
                return
            else:
                print("Checksum does not match, downloading dataset.")

        with urlopen(DATASET_URL, timeout=30) as resp:
            data = resp.read()

        with ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(DATA_DIR_RAW)

    def __load_labels(path: Path):
        with path.open('rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())
        return labels, size

    def __load_images(path: Path):
        with path.open('rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        return image_data, size, rows, cols

    def format():
        images_filepath_train = DATA_DIR_RAW / 'train-images-idx3-ubyte/train-images-idx3-ubyte'
        labels_filepath_train = DATA_DIR_RAW / 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
        images_filepath_test = DATA_DIR_RAW / 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
        labels_filepath_test = DATA_DIR_RAW / 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

        labels = array("B")
        labels_size = 0
        for filepath in (labels_filepath_train, labels_filepath_test):
            labels_loaded, size = MNIST.__load_labels(filepath)
            labels.extend(labels_loaded)
            labels_size += size

        image_data = array("B")
        images_size = 0
        for filepath in (images_filepath_train, images_filepath_test):
            images_loaded, size, rows, cols = MNIST.__load_images(filepath)
            image_data.extend(images_loaded)
            images_size += size

        assert labels_size == images_size, f"Dataset sizes do not match: {images_size} images != {labels_size} labels"

        DATA_DIR_FORMATTED_LABELS.parent.mkdir(exist_ok=True)
        with DATA_DIR_FORMATTED_LABELS.open('wb') as file:
            file.write(struct.pack(">II", 2049, len(labels)))
            file.write(labels.tobytes())

        DATA_DIR_FORMATTED_IMAGES.parent.mkdir(exist_ok=True)
        with DATA_DIR_FORMATTED_IMAGES.open('wb') as file:
            file.write(struct.pack(">IIII", 2051, images_size, rows, cols))
            file.write(image_data.tobytes())

    def load() -> Dataset:
        label_data, _ = MNIST.__load_labels(DATA_DIR_FORMATTED_LABELS)
        image_data, _, rows, cols = MNIST.__load_images(DATA_DIR_FORMATTED_IMAGES)

        labels = np.frombuffer(label_data, dtype=np.uint8).copy()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(-1, rows, cols).copy()
        return Dataset(images=images, labels=labels)


ROTATION_MAX_DEGREES = 50.0
ROTATION_STD_DEGREES = ROTATION_MAX_DEGREES / 3  # ~99.7% of draws fall within the limit before clipping
TRANSLATION_STD_PIXELS = 2.0
BRIGHTNESS_MIN_SCALE = 0.5
BRIGHTNESS_MAX_SCALE = 1.5

_rng = np.random.default_rng()

def Rotation(source_dataset: Dataset, rng: np.random.Generator = _rng) -> Dataset:
    n = len(source_dataset.images)
    degrees = np.clip(rng.standard_normal(n) * ROTATION_STD_DEGREES, -ROTATION_MAX_DEGREES, ROTATION_MAX_DEGREES)
    images = np.empty_like(source_dataset.images)
    for i, (image, angle) in enumerate(zip(source_dataset.images, degrees)):
        ndimage.rotate(image, angle, reshape=False, order=1, mode="constant", cval=0, output=images[i])
    return Dataset(images=images, labels=source_dataset.labels.copy())

def Translation(source_dataset: Dataset, rng: np.random.Generator = _rng) -> Dataset:
    n = len(source_dataset.images)
    # Whole-pixel shifts keep digits sharp; extreme draws may clip the digit at the border
    offsets = np.rint(rng.standard_normal((n, 2)) * TRANSLATION_STD_PIXELS)
    images = np.empty_like(source_dataset.images)
    for i, (image, offset) in enumerate(zip(source_dataset.images, offsets)):
        ndimage.shift(image, offset, order=0, mode="constant", cval=0, output=images[i])
    return Dataset(images=images, labels=source_dataset.labels.copy())

def Brightness(source_dataset: Dataset, rng: np.random.Generator = _rng) -> Dataset:
    n = len(source_dataset.images)
    scale = rng.uniform(BRIGHTNESS_MIN_SCALE, BRIGHTNESS_MAX_SCALE, size=(n, 1, 1))
    images = np.clip(np.rint(source_dataset.images * scale), 0, 255).astype(source_dataset.images.dtype)
    return Dataset(images=images, labels=source_dataset.labels.copy())

def get_dataset(*transformations):
    MNIST.download()
    MNIST.format()
    source_dataset = MNIST.load()
    dest_dataset = Dataset.empty(*source_dataset.images.shape[1:])
    dest_dataset.extend(source_dataset)

    for transform in transformations:
        dest_dataset.extend(transform(source_dataset))

    return dest_dataset

def main():
    # cli for interacting with datasets
    # use argparse
    # options to download/redownload the dataset
    # generate different translations
    print("Hello, from data.")
    
    out = get_dataset(Rotation)

    for image_a, image_b in zip(out.images[80000:80010], out.images[10000:10010]):
        for image in (image_a, image_b):
            for x in image:
                for y in x:
                    print([' ','.','"','-','+','o','O','M','#','@'][int(y/256*10)], end="")
                print()

if __name__ == "__main__":
    main()
