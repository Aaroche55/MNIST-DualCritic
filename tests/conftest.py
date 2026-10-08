import io
import struct
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from mnist_dualcritic import data
from mnist_dualcritic.data import SourceDataset

ROWS = COLS = 12


def make_split(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic images: a random blob kept away from the border, like a centred digit."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, ROWS, COLS), dtype=np.uint8)
    images[:, 3:9, 4:8] = rng.integers(1, 256, size=(n, 6, 4), dtype=np.uint8)
    labels = rng.integers(0, 10, size=n, dtype=np.uint8)
    return images, labels


def idx_images_bytes(images: np.ndarray) -> bytes:
    # Encoded independently of the code under test, straight from the IDX spec
    return struct.pack(">IIII", 2051, *images.shape) + images.tobytes()


def idx_labels_bytes(labels: np.ndarray) -> bytes:
    return struct.pack(">II", 2049, len(labels)) + labels.tobytes()


TRAIN = make_split(seed=1, n=6)
TEST = make_split(seed=2, n=4)
RAW_FILES = {
    "train/images.idx3": idx_images_bytes(TRAIN[0]),
    "train/labels.idx1": idx_labels_bytes(TRAIN[1]),
    "test/images.idx3": idx_images_bytes(TEST[0]),
    "test/labels.idx1": idx_labels_bytes(TEST[1]),
}


class FakeDataset(SourceDataset):
    name = "fake"
    url = "https://example.invalid/fake.zip"
    raw_files = (("train/images.idx3", "train/labels.idx1"), ("test/images.idx3", "test/labels.idx1"))
    checksums: ClassVar[dict[str, str]] = {path: sha256(content).hexdigest() for path, content in RAW_FILES.items()}


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if anything tries a real download; tests that need one install a fake server."""

    def refuse(url, timeout=None):
        raise AssertionError(f"Unexpected network access: {url}")

    monkeypatch.setattr(data, "urlopen", refuse)


@pytest.fixture
def serve_zip(monkeypatch):
    """Serve the given files as the download; returns a list recording each requested URL."""
    requests: list[str] = []

    def install(files: dict[str, bytes] = RAW_FILES):
        payload = zip_bytes(files)

        def fake_urlopen(url, timeout=None):
            requests.append(url)
            return io.BytesIO(payload)

        monkeypatch.setattr(data, "urlopen", fake_urlopen)
        return requests

    return install


@pytest.fixture
def data_dir(tmp_path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def raw_source(data_dir) -> FakeDataset:
    """A fake dataset whose raw files are already 'downloaded'."""
    source = FakeDataset(data_dir)
    for path, content in RAW_FILES.items():
        (source.raw_dir / path).parent.mkdir(parents=True, exist_ok=True)
        (source.raw_dir / path).write_bytes(content)
    return source


@pytest.fixture
def source(raw_source) -> FakeDataset:
    """A fake dataset that has also been formatted."""
    raw_source.format()
    return raw_source
