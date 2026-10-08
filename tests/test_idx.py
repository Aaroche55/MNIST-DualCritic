import struct

import numpy as np
import pytest

from mnist_dualcritic.data import read_idx_images, read_idx_labels, write_idx_images, write_idx_labels

from .conftest import TRAIN, idx_images_bytes, idx_labels_bytes


def test_reads_spec_encoded_files(tmp_path):
    images, labels = TRAIN
    (tmp_path / "images").write_bytes(idx_images_bytes(images))
    (tmp_path / "labels").write_bytes(idx_labels_bytes(labels))

    np.testing.assert_array_equal(read_idx_images(tmp_path / "images"), images)
    np.testing.assert_array_equal(read_idx_labels(tmp_path / "labels"), labels)


def test_write_matches_spec_encoding(tmp_path):
    images, labels = TRAIN
    write_idx_images(tmp_path / "nested" / "images", images)
    write_idx_labels(tmp_path / "nested" / "labels", labels)

    assert (tmp_path / "nested" / "images").read_bytes() == idx_images_bytes(images)
    assert (tmp_path / "nested" / "labels").read_bytes() == idx_labels_bytes(labels)


def test_read_returns_writable_arrays(tmp_path):
    write_idx_images(tmp_path / "images", TRAIN[0])
    images = read_idx_images(tmp_path / "images")
    images[0, 0, 0] = 1  # np.frombuffer alone would give a read-only view


@pytest.mark.parametrize(
    ("reader", "content"),
    [
        (read_idx_images, struct.pack(">IIII", 2049, 1, 1, 1) + b"\x00"),
        (read_idx_labels, idx_images_bytes(TRAIN[0])),
    ],
    ids=["images", "labels"],
)
def test_rejects_wrong_magic_number(tmp_path, reader, content):
    (tmp_path / "file").write_bytes(content)
    with pytest.raises(ValueError, match="Magic number mismatch"):
        reader(tmp_path / "file")


@pytest.mark.parametrize("reader", [read_idx_images, read_idx_labels])
def test_rejects_truncated_header(tmp_path, reader):
    (tmp_path / "file").write_bytes(b"\x00\x00\x08")
    with pytest.raises(ValueError, match="too short to contain an IDX header"):
        reader(tmp_path / "file")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (lambda images: images.tobytes(), "header says 7 images, found 6"),
        (lambda images: images.tobytes()[:-5], "header says 7 images, found 5.96528"),
    ],
    ids=["missing image", "truncated image"],
)
def test_rejects_body_not_matching_header(tmp_path, body, message):
    images = TRAIN[0]
    header = struct.pack(">IIII", 2051, len(images) + 1, *images.shape[1:])
    (tmp_path / "images").write_bytes(header + body(images))
    with pytest.raises(ValueError, match=message):
        read_idx_images(tmp_path / "images")


@pytest.mark.parametrize("writer", [write_idx_images, write_idx_labels])
def test_refuses_non_uint8(tmp_path, writer):
    array = np.full((1, 2, 2), 300.0) if writer is write_idx_images else np.array([300])
    with pytest.raises(TypeError, match="must be uint8"):
        writer(tmp_path / "file", array)
    assert not (tmp_path / "file").exists()
