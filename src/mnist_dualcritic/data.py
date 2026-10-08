import struct
import io
import numpy as np
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
    images: list
    labels: list

    def extend(self, other_dataset: Self):
        self.images.extend(other_dataset.images)
        self.labels.extend(other_dataset.labels)

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

    def format():
        images_filepath_train = DATA_DIR_RAW / 'train-images-idx3-ubyte/train-images-idx3-ubyte'
        labels_filepath_train = DATA_DIR_RAW / 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
        images_filepath_test = DATA_DIR_RAW / 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
        labels_filepath_test = DATA_DIR_RAW / 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

        labels = array("B")
        for filepath in (labels_filepath_train, labels_filepath_test):
            with filepath.open('rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                if magic != 2049:
                    raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
                labels.extend(array("B", file.read()))
        
        image_data = array("B")
        for filepath in (images_filepath_train, images_filepath_test):
            with filepath.open('rb') as file:
                magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
                if magic != 2051:
                    raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
                image_data.extend(array("B", file.read()))
        
        DATA_DIR_FORMATTED_LABELS.parent.mkdir(exist_ok=True)
        with DATA_DIR_FORMATTED_LABELS.open('wb') as file:
            file.write(struct.pack(">II", 2049, len(labels)))
            file.write(labels.tobytes())

        DATA_DIR_FORMATTED_IMAGES.parent.mkdir(exist_ok=True)
        with DATA_DIR_FORMATTED_IMAGES.open('wb') as file:
            file.write(struct.pack(">IIII", 2051, size, rows, cols))
            file.write(image_data.tobytes())

    def load():
        with DATA_DIR_FORMATTED_LABELS.open('rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())
        with DATA_DIR_FORMATTED_IMAGES.open('rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        images = []
        for _ in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            

        return Dataset(images=images, labels=labels.tolist())


def Rotation(source_dataset: Dataset) -> Dataset: return Dataset([], [])
def Translation(source_dataset: Dataset) -> Dataset: return Dataset([], [])
def Brightness(source_dataset: Dataset) -> Dataset: return Dataset([], [])

def get_dataset(*transformations):
    MNIST.download()
    MNIST.format()
    source_dataset = MNIST.load()
    dest_dataset = Dataset([], [])
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

    print(out.labels[:1000])
    for image in out.images[:1]:
        for x in image:
            for y in x:
                print([' ','.','"','-','+','o','O','M','#','@'][int(y/256*10)], end="")
            print()

if __name__ == "__main__":
    main()
